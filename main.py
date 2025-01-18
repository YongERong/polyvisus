import cv2
import mediapipe as mp
from typing import Tuple
from collections import defaultdict
import time

class AdaptiveVirtualKeyboard:
    def __init__(self):
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
        self.current_layout = "letters"
        self.shift_active = False
        self.key_size = 40
        self.pressed_keys = set()
        self.typed_text = ""
        self.last_press_time = 0
        self.key_cooldown = 0.5

        # Combined finger tracking
        self.prev_tip_positions = {}  # Track previous tip positions
        self.prev_tip_y_velocity = {}  # Track y-velocity of tips
        self.movement_threshold = 0.01  # Minimum movement to consider
        self.press_velocity_threshold = 0.04  # Minimum velocity for a press
        self.x_velocity_max_ratio = 1.5  # Maximum allowed ratio of x-velocity to y-velocity
        self.min_consecutive_downward = 2
        self.tip_history = defaultdict(list)  # Store recent tip positions
        self.history_length = 3  # Number of frames to track
        
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
        self.layout_update_interval = 10
        self.init_layouts()

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
            if len(self.tip_history[finger_key]) > self.history_length:
                self.tip_history[finger_key].pop(0)
            
            # Need full history for velocity calculations
            if len(self.tip_history[finger_key]) < self.history_length:
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
            
            if (downward_frames >= self.min_consecutive_downward and
                avg_y_velocity > self.press_velocity_threshold and
                avg_x_velocity < avg_y_velocity * self.x_velocity_max_ratio and
                total_movement > self.movement_threshold):
                active_fingers.append((finger_name, tip.x, tip.y))
        
        return active_fingers


    def check_key_press(self, finger_pos: Tuple[int, int]) -> str:
        for letter, key_pos in self.current_keys.items():
            if (abs(finger_pos[0] - key_pos[0]) < self.key_size//2 and
                abs(finger_pos[1] - key_pos[1]) < self.key_size//2):
                return letter
        return ""

    def init_layouts(self):
        # Base positions for different layouts
        self.layout_configs = {
            "letters": {
                # Top row (furthest from user)
                'Q': (100, 300), 'W': (200, 300), 'E': (300, 300), 'R': (400, 300),
                'T': (500, 300), 'Y': (600, 300), 'U': (700, 300), 'I': (800, 300),
                'O': (900, 300), 'P': (1000, 300),
                
                # Middle row
                'A': (150, 200), 'S': (250, 200), 'D': (350, 200), 'F': (450, 200),
                'G': (550, 200), 'H': (650, 200), 'J': (750, 200), 'K': (850, 200),
                'L': (950, 200),
                
                # Bottom row
                'Z': (200, 100), 'X': (300, 100), 'C': (400, 100), 'V': (500, 100),
                'B': (600, 100), 'N': (700, 100), 'M': (800, 100),
                
                # Special keys (closest to user)
                'SHIFT': (100, 50), '123': (250, 50), 'SPACE': (500, 50),
                'BACK': (750, 50), 'ENTER': (900, 50)
            },
            "numbers": {
                # Top row
                '1': (100, 300), '2': (200, 300), '3': (300, 300), '4': (400, 300),
                '5': (500, 300), '6': (600, 300), '7': (700, 300), '8': (800, 300),
                '9': (900, 300), '0': (1000, 300),
                
                # Middle row
                '@': (150, 200), '#': (250, 200), '$': (350, 200), '%': (450, 200),
                '&': (550, 200), '*': (650, 200), '-': (750, 200), '+': (850, 200),
                '=': (950, 200),
                
                # Bottom row
                '(': (200, 100), ')': (300, 100), '[': (400, 100), ']': (500, 100),
                '{': (600, 100), '}': (700, 100), '|': (800, 100),
                
                # Special keys
                'ABC': (100, 50), 'SYM': (250, 50), 'SPACE': (500, 50),
                'BACK': (750, 50), 'ENTER': (900, 50)
            },
            "symbols": {
                # Top row
                '!': (100, 300), '?': (200, 300), ',': (300, 300), '.': (400, 300),
                ';': (500, 300), ':': (600, 300), '"': (700, 300), "'": (800, 300),
                '/': (900, 300), '\\': (1000, 300),
                
                # Middle row
                '~': (150, 200), '`': (250, 200), '^': (350, 200), '_': (450, 200),
                '<': (550, 200), '>': (650, 200), '€': (750, 200), '£': (850, 200),
                '¥': (950, 200),
                
                # Special keys
                'ABC': (100, 50), '123': (250, 50), 'SPACE': (500, 50),
                'BACK': (750, 50), 'ENTER': (900, 50)
            }
        }
        self.current_keys = self.layout_configs["letters"].copy()

    def update_layout(self):
        """Update key positions based on usage patterns"""
        current_time = time.time()
        if current_time - self.last_layout_update < self.layout_update_interval:
            return

        # Remove old press history
        current_time = time.time()
        self.press_history = [press for press in self.press_history 
                            if current_time - press[1] < 60]  # Keep last minute

        if len(self.press_history) < 10:  # Need minimum data points
            return

        # Calculate average position for frequently used keys
        key_positions = defaultdict(list)
        for key, pos, _ in self.press_history:
            key_positions[key].append(pos)

        # Update positions for frequently used keys
        base_layout = self.layout_configs[self.current_layout]
        new_layout = base_layout.copy()

        for key, positions in key_positions.items():
            if len(positions) >= 5:  # Minimum presses to consider adjustment
                avg_x = sum(p[0] for p in positions) / len(positions)
                avg_y = sum(p[1] for p in positions) / len(positions)
                
                # Limit movement from original position
                orig_x, orig_y = base_layout[key]
                max_movement = 20
                new_x = min(max(avg_x, orig_x - max_movement), orig_x + max_movement)
                new_y = min(max(avg_y, orig_y - max_movement), orig_y + max_movement)
                
                new_layout[key] = (int(new_x), int(new_y))

        self.current_keys = new_layout
        self.last_layout_update = current_time

    def handle_special_keys(self, key: str) -> bool:
        """Handle special key presses"""
        if key == 'SHIFT':
            self.shift_active = not self.shift_active
            return True
        elif key in ['123', 'ABC', 'SYM']:
            if key == '123':
                self.current_layout = "numbers"
            elif key == 'ABC':
                self.current_layout = "letters"
            elif key == 'SYM':
                self.current_layout = "symbols"
            self.current_keys = self.layout_configs[self.current_layout].copy()
            return True
        elif key == 'SPACE':
            self.typed_text += " "
            return True
        elif key == 'BACK':
            self.typed_text = self.typed_text[:-1]
            return True
        elif key == 'ENTER':
            self.typed_text += "\n"
            return True
        return False

    def draw_keyboard(self, frame):
        # Draw each key
        for letter, pos in self.current_keys.items():
            # Determine key color
            is_pressed = letter in self.pressed_keys
            is_special = letter in ['SHIFT', '123', 'ABC', 'SYM', 'SPACE', 'BACK', 'ENTER']
            
            if is_pressed:
                color = (0, 255, 0)
            elif is_special:
                color = (0, 165, 255)
            else:
                color = (255, 255, 255)

            # Draw key background
            cv2.rectangle(frame, 
                         (pos[0] - self.key_size//2, pos[1] - self.key_size//2),
                         (pos[0] + self.key_size//2, pos[1] + self.key_size//2),
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
        cv2.rectangle(frame, (50, 270), (600, 320), (0, 0, 0), -1)
        cv2.putText(frame, self.typed_text[-40:],  # Show last 40 characters
                   (50, 300),
                   cv2.FONT_HERSHEY_SIMPLEX, 1,
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
            if len(self.tip_history[finger_key]) > self.history_length:
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
            if (movement > self.movement_threshold and 
                y_velocity > 0 and 
                y_velocity > self.press_velocity_threshold):
                active_fingers.append((finger_name, tip.x, tip.y))
        
        return active_fingers

    def process_frame(self, frame):
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
                        if current_time - self.last_press_time > self.key_cooldown:
                            if not self.handle_special_keys(pressed_key):
                                # Add regular key to typed text
                                if len(pressed_key) == 1:  # Regular character
                                    if self.shift_active:
                                        self.typed_text += pressed_key.upper()
                                    else:
                                        self.typed_text += pressed_key.lower()
                            
                            # Record press for layout adaptation
                            self.press_history.append((pressed_key, current_time, (fx, fy)))
                            self.last_press_time = current_time
        
        # Draw the keyboard
        self.draw_keyboard(frame)
        
        return frame

def main():
    cap = cv2.VideoCapture(0)
    keyboard = AdaptiveVirtualKeyboard()
    
    while cap.isOpened():
        success, frame = cap.read()
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
