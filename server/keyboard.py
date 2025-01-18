import cv2
import mediapipe as mp
from typing import Tuple
from collections import defaultdict
import time
import numpy as np
import json
import sys
#sys.path.append('../spellchecker')
from spellchecker import spellchecker
import random

class AdaptiveVirtualKeyboard:
    def __init__(self):
        # Initialize spellchecker
        self.spell = spellchecker

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize keyboard state
        self.config = json.load(open('server/config.json'))
        self.settings = self.config['settings']
        self.current_layout = "letters"
        self.shift_active = False
        self.pressed_keys = set()
        self.typed_text = ""
        self.last_press_time = 0
        self.challenges = json.load(open('server/challenges.json'))
        self.current_challenge = random.choice(self.challenges[self.settings["difficulty"]])
        # Combined finger tracking
        self.prev_tip_positions = {}  # Track previous tip positions
        self.prev_tip_y_velocity = {}  # Track y-velocity of tips
        self.tip_history = defaultdict(list)  # Store recent tip positions
        
        # Define fingertip landmarks
        self.fingertips = {
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'ring': 16,
            'pinky': 20
        }
        
        # Rest of initialization remains the same
        self.heat_map = defaultdict(int)
        self.press_history = []
        self.last_layout_update = time.time()
        self.init_layout(self.config['layouts'])
        
        # Add metrics tracking
        self.char_timestamps = []
        self.metrics_update_time = time.time()
        self.current_wpm = 0
        self.current_spc = 0
        
        
    def detect_finger_press(self, hand_landmarks, hand_id) -> list:
        """
        Detect deliberate downward pressing motions while ignoring quick horizontal movements
        """
        active_fingers = []
        
        for finger_name, tip_idx in self.fingertips.items():
            # Get current tip position
            tip = hand_landmarks.landmark[tip_idx]
            
            # Create unique key for this finger
            finger_key = f"{hand_id}_{finger_name}"
            
            # Store current position in history
            self.tip_history[finger_key].append((tip.x, tip.y))
            if len(self.tip_history[finger_key]) > self.settings["history_length"]:
                self.tip_history[finger_key].pop(0)
            
            # Need full history for velocity calculations
            if len(self.tip_history[finger_key]) < self.settings["history_length"]:
                continue
                
            # Calculate velocities using multiple frames
            y_velocities = []
            x_velocities = []
            for i in range(1, len(self.tip_history[finger_key])):
                current_pos = self.tip_history[finger_key][i]
                prev_pos = self.tip_history[finger_key][i-1]
                
                y_velocities.append(current_pos[1] - prev_pos[1])
                x_velocities.append(abs(current_pos[0] - prev_pos[0]))
            
            # Check for consistent downward motion
            downward_frames = sum(1 for v in y_velocities if v > 0)
            
            # Calculate average velocities
            avg_y_velocity = sum(y_velocities) / len(y_velocities)
            avg_x_velocity = sum(x_velocities) / len(x_velocities)
            
            # Conditions for a valid press:
            # 1. Consistent downward motion over multiple frames
            # 2. Average y-velocity exceeds threshold
            # 3. Horizontal movement is not too large compared to vertical movement
            # 4. Total movement exceeds minimum threshold
            total_movement = ((tip.x - self.tip_history[finger_key][0][0])**2 + 
                            (tip.y - self.tip_history[finger_key][0][1])**2)**0.5
            
            if (downward_frames >= self.settings["min_consecutive_downward"] and
                avg_y_velocity > self.settings["press_velocity_threshold"] and
                avg_x_velocity < avg_y_velocity * self.settings["x_velocity_max_ratio"] and
                total_movement > self.settings["movement_threshold"]):
                active_fingers.append((finger_name, tip.x, tip.y))
        
        return active_fingers


    def check_key_press(self, finger_pos: Tuple[int, int]) -> str:
        for letter, key_pos in self.current_keys.items():
            if (abs(finger_pos[0] - key_pos[0]) < self.settings["key_size"]//2 and
                abs(finger_pos[1] - key_pos[1]) < self.settings["key_size"]//2):
                return letter
        return ""

    def init_layout(self, layouts):
        def get_relative_pos(x_percent, y_percent, screen_width=1000, screen_height=400):
            """Convert percentage positions to pixel coordinates"""
            return (int(x_percent * screen_width), int(y_percent * screen_height))

        def auto_layout(layout):
            formatted_layout = {}
            n_rows = len(layout)
            v_margin = self.settings["vertical_margin"]
            h_margin = self.settings["horizontal_margin"]
            
            # Pre-calculate y positions for each row
            center = 0.5
            key_spacing = self.settings["key_size"] / 400  # Convert pixels to ratio
            y_positions = [center + (i - (n_rows-1)/2) * key_spacing for i in range(n_rows)]
            
            for i, row in enumerate(layout):
                n_cols = len(row)
                y_ratio = y_positions[i]
                
                # Pre-calculate x positions for this row
                x_positions = [h_margin + j * (1 - 2*h_margin)/(n_cols-1) for j in range(n_cols)]
                
                # Batch process each row
                formatted_layout.update({
                    letter: get_relative_pos(x_positions[j], y_ratio)
                    for j, letter in enumerate(row)
                })
                
            return formatted_layout

        # Process all layouts at once
        self.layout_configs = {
            name: auto_layout(layout) 
            for name, layout in layouts.items()
        }

        self.current_keys = self.layout_configs["letters"].copy()

    def improve_layout(self):
        """
        Update key positions based on usage patterns to minimize finger movement.
        Keys that are frequently used together will be positioned closer to each other.
        """
        current_time = time.time()
        if current_time - self.last_layout_update < self.settings["layout_update_interval"]:
            return

        # Remove old press history (keep last 2 minutes)
        self.press_history = [press for press in self.press_history 
                             if current_time - press[1] < 120]

        if len(self.press_history) < 10:  # Need minimum data points
            return

        # Analyze key transitions
        key_transitions = defaultdict(list)
        for i in range(len(self.press_history) - 1):
            current_key = self.press_history[i][0]
            next_key = self.press_history[i + 1][0]
            if current_key not in ['SHIFT', '123', 'ABC', 'SYM', 'SPACE', 'BACK', 'CLR'] and \
               next_key not in ['SHIFT', '123', 'ABC', 'SYM', 'SPACE', 'BACK', 'CLR']:
                key_transitions[current_key].append(next_key)

        # Calculate optimal positions based on transitions
        base_layout = self.layout_configs[self.current_layout]
        new_layout = base_layout.copy()
        
        for key, transitions in key_transitions.items():
            if len(transitions) >= 3:  # Minimum transitions to consider adjustment
                # Calculate the average position of frequently following keys
                following_positions = []
                for next_key in transitions:
                    if next_key in new_layout:
                        following_positions.append(new_layout[next_key])
                
                if following_positions:
                    # Calculate target position (weighted average between original and optimal)
                    orig_x, orig_y = base_layout[key]
                    target_x = sum(p[0] for p in following_positions) / len(following_positions)
                    target_y = sum(p[1] for p in following_positions) / len(following_positions)
                    
                    # Apply weighted movement (30% toward optimal position)
                    weight = 0.3
                    max_movement = self.settings["key_size"]  # Limit movement to key size
                    
                    new_x = orig_x + min(max(target_x - orig_x, -max_movement), max_movement) * weight
                    new_y = orig_y + min(max(target_y - orig_y, -max_movement), max_movement) * weight
                    
                    # Ensure keys don't overlap
                    min_spacing = self.settings["key_size"] * 1.2
                    for other_key, other_pos in new_layout.items():
                        if other_key != key:
                            dx = new_x - other_pos[0]
                            dy = new_y - other_pos[1]
                            distance = (dx**2 + dy**2)**0.5
                            if distance < min_spacing:
                                # Push keys apart if too close
                                scale = min_spacing / (distance + 1e-6)
                                new_x = other_pos[0] + dx * scale
                                new_y = other_pos[1] + dy * scale
                    
                    new_layout[key] = (int(new_x), int(new_y))

        self.current_keys = new_layout
        self.last_layout_update = current_time
    
    def deprove_layout(self):
        """
        De-improve layout by randomly swapping key positions while maintaining original position bounds
        """
        # Get current layout keys and positions
        base_layout = self.layout_configs[self.current_layout]
        keys = list(base_layout.keys())
        positions = list(base_layout.values())
        
        # Randomly select two keys to swap positions
        if len(keys) >= 2:
            idx1, idx2 = random.sample(range(len(keys)), 2)
            positions[idx1], positions[idx2] = positions[idx2], positions[idx1]
            
            # Create new layout with swapped positions
            new_layout = {k: p for k, p in zip(keys, positions)}
            
            # Update both current keys and base layout config
            self.current_keys = new_layout.copy()
            self.layout_configs[self.current_layout] = new_layout.copy()

    def handle_special_keys(self, key: str) -> bool:
        """Handle special key presses"""
        def update_layout():
            self.current_keys = self.layout_configs[self.current_layout].copy()
            return True
        
        def spell_check():
            current_word = self.get_current_word()
            corrected_word = self.spell.correction(current_word) 
            if corrected_word != current_word:
                self.typed_text = self.typed_text[:-len(current_word)] + corrected_word  # Replace with corrected word
            self.typed_text += " "
            return True
        
        match key:
            case 'SHIFT':
                self.shift_active = not self.shift_active
                return True
            case '123':
                self.current_layout = "numbers"
                update_layout()
            case 'ABC':
                self.current_layout = "letters"
                update_layout()
            case 'SYM':
                self.current_layout = "symbols"
                update_layout()
            case 'SPACE':
                if self.settings["difficulty"] == 0:
                    spell_check()
                    return True
                else:
                    self.typed_text += " "
            case 'BACK':
                self.typed_text = self.typed_text[:-1]
                return True
            case 'CLR':
                self.typed_text = ""
                return True
            case _:
                return False
    
    def get_current_word(self) -> str:
        """Get the current word being typed, assuming space or punctuation breaks it"""
        words = self.typed_text.split()
        return words[-1] if words else ""


    def draw_keyboard(self, frame):
        for letter, pos in self.current_keys.items():
            # Determine key color
            is_pressed = letter in self.pressed_keys
            is_special = letter in ['SHIFT', '123', 'ABC', 'SYM', 'SPACE', 'BACK', 'CLR']
            
            if is_pressed:
                color = (0, 255, 0)
            elif is_special:
                color = (0, 165, 255)
            else:
                color = (255, 255, 255)

            # Draw key background
            cv2.rectangle(frame, 
                        (pos[0] - self.settings["key_size"]//2, pos[1] - self.settings["key_size"]//2),
                        (pos[0] + self.settings["key_size"]//2, pos[1] + self.settings["key_size"]//2),
                        color, 2)
            
            # Draw key label
            display_text = letter
            if self.shift_active and len(letter) == 1 and letter.isalpha():
                display_text = letter.upper()
            elif not self.shift_active and len(letter) == 1 and letter.isalpha():
                display_text = letter.lower()

            # Adjust text size for special keys
            font_scale = 0.4 if len(display_text) > 1 else 0.8
            text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = pos[0] - text_size[0]//2
            text_y = pos[1] + text_size[1]//2

            
            
            cv2.putText(frame, display_text,
                       (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       color, 2)
        
        # Draw typed text
        cv2.rectangle(frame, (50, frame.shape[0] - 80), (600, frame.shape[0] - 30), (0, 0, 0), -1)
        cv2.putText(frame, self.typed_text[-40:],  # Show last 40 characters
                   (50, frame.shape[0] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1,
                   (255, 255, 255), 2)
        display_height = 100
        display_width = int(frame.shape[1] * 0.8)  # 80% of frame width
        display_x = (frame.shape[1] - display_width) // 2  # Center horizontally
        display_y = 30
            
        cv2.rectangle(frame, 
                    (display_x, display_y),
                    (display_x + display_width, display_y + display_height),
                    (255, 255, 255), 2)  # White outline
        
        # Calculate text size to center it
        text_size = cv2.getTextSize(self.current_challenge, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = display_x + (display_width - text_size[0]) // 2  # Center horizontally
        text_y = display_y + (display_height + text_size[1]) // 2  # Center vertically
        
        cv2.putText(frame, self.current_challenge,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

        # Draw metrics in top right corner
        metrics_text = f"WPM: {self.current_wpm:.1f} | SPC: {self.current_spc:.2f}s"
        font_scale = 0.7
        text_size = cv2.getTextSize(metrics_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        text_x = frame.shape[1] - text_size[0] - 20  # 20 pixels from right edge
        text_y = text_size[1] + 20  # 20 pixels from top
        
        # Draw background rectangle
        padding = 10
        cv2.rectangle(frame, 
                     (text_x - padding, text_y - text_size[1] - padding),
                     (text_x + text_size[0] + padding, text_y + padding),
                     (0, 0, 0), -1)
        
        # Draw metrics text
        cv2.putText(frame, metrics_text,
                   (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                   (255, 255, 255), 2)

    def detect_finger_press(self, hand_landmarks, hand_id) -> list:
        """
        Detect deliberate pressing motions using both position and velocity
        """
        active_fingers = []
        
        for finger_name, tip_idx in self.fingertips.items():
            # Get current tip position
            tip = hand_landmarks.landmark[tip_idx]
            
            # Create unique key for this finger
            finger_key = f"{hand_id}_{finger_name}"
            
            # Store current position in history
            self.tip_history[finger_key].append((tip.x, tip.y))
            if len(self.tip_history[finger_key]) > self.settings["history_length"]:
                self.tip_history[finger_key].pop(0)
            
            # Need at least 2 points to calculate velocity
            if len(self.tip_history[finger_key]) < 2:
                continue
                
            # Calculate recent movement
            current_pos = self.tip_history[finger_key][-1]
            prev_pos = self.tip_history[finger_key][-2]
            
            # Calculate total movement and y-velocity
            movement = ((current_pos[0] - prev_pos[0])**2 + 
                    (current_pos[1] - prev_pos[1])**2)**0.5
            y_velocity = current_pos[1] - prev_pos[1]
            
            # Store velocity for the next frame
            self.prev_tip_y_velocity[finger_key] = y_velocity
            
            # Check for deliberate press:
            # 1. Movement exceeds minimum threshold
            # 2. Y-velocity indicates downward movement (positive in screen coordinates)
            # 3. Y-velocity exceeds press threshold
            if (movement > self.settings["movement_threshold"] and 
                y_velocity > 0 and 
                y_velocity > self.settings["press_velocity_threshold"]):
                active_fingers.append((finger_name, tip.x, tip.y))
        
        return active_fingers
    # Update key positions based on current frame dimensions
    def update_positions(self, frame):
        h, w, c = frame.shape
        self.current_keys = {k: (int(pos[0] * w/1000), int(pos[1] * h/400)) 
                for k, pos in self.layout_configs[self.current_layout].items()}
    


    def process_frame(self, frame):
        h, w, c = frame.shape   
        
    
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        self.pressed_keys.clear()
        current_time = time.time()
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Get all active fingertips
                active_fingers = self.detect_finger_press(hand_landmarks, hand_idx)
                
                h, w, c = frame.shape
                
                # Process each active finger
                for finger_name, fx_norm, fy_norm in active_fingers:
                    # Convert normalized coordinates to pixel coordinates
                    fx = int(fx_norm * w)
                    fy = int(fy_norm * h)
                    
                    # Draw fingertip position
                    cv2.circle(frame, (fx, fy), 5, (0, 255, 0), -1)
                    
                    # Check for key presses
                    pressed_key = self.check_key_press((fx, fy))
                    if pressed_key:
                        self.pressed_keys.add(pressed_key)
                        if current_time - self.last_press_time > self.settings["key_cooldown"]:
                            if not self.handle_special_keys(pressed_key):
                                # Add regular key to typed text
                                if len(pressed_key) == 1:  # Regular character
                                    if self.shift_active:
                                        self.typed_text += pressed_key.upper()
                                    else:
                                        self.typed_text += pressed_key.lower()
                                    # Add timestamp for metrics
                                    self.char_timestamps.append(time.time())
                            
                            # Record press for layout adaptation
                            self.press_history.append((pressed_key, current_time, (fx, fy)))
                            self.last_press_time = current_time

                            if self.settings["difficulty"] == 0:
                                self.improve_layout()
                            elif self.settings["difficulty"] == 2:
                                if len(self.typed_text) % random.randint(4, 5) == 0:
                                    self.deprove_layout()

        
        # Update metrics before drawing
        self.update_metrics()
        
        # Draw the keyboard
        self.draw_keyboard(frame)
        
        return frame

    def update_metrics(self):
        """Calculate and update WPM and SPC metrics"""
        current_time = time.time()
        
        # Only update every 0.5 seconds to avoid excessive calculations
        if current_time - self.metrics_update_time < 0.5:
            return
            
        # Remove timestamps older than 60 seconds
        cutoff_time = current_time - 60
        self.char_timestamps = [t for t in self.char_timestamps if t > cutoff_time]
        
        if len(self.char_timestamps) < 2:
            self.current_wpm = 0
            self.current_spc = 0
            return
            
        # Calculate WPM
        # Standard assumption: 5 characters = 1 word
        char_count = len(self.char_timestamps)
        time_span = self.char_timestamps[-1] - self.char_timestamps[0]
        if time_span > 0:
            chars_per_minute = (char_count / time_span) * 60
            self.current_wpm = chars_per_minute / 5  # Convert to words per minute
            self.current_spc = time_span / char_count
        
        self.metrics_update_time = current_time

def main():
    cap = cv2.VideoCapture(0)
    keyboard = AdaptiveVirtualKeyboard()
    
    while cap.isOpened():
        success, frame = cap.read()
        keyboard.update_positions(frame)
        
        if not success:
            continue
        
        frame = keyboard.process_frame(frame)
        cv2.imshow('Adaptive Virtual Keyboard', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Final text:", keyboard.typed_text)
            break
        else:
            print("Current text:", keyboard.typed_text)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
