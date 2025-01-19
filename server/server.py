import cv2
import numpy as np
from flask import Flask, Response, jsonify
import threading
from keyboard import AdaptiveVirtualKeyboard  # Your existing keyboard class
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class KeyboardServer:
    def __init__(self):
        self.keyboard = AdaptiveVirtualKeyboard()
        self.cap = cv2.VideoCapture(0)
        self.current_text = ""
        self.keyboard_state = {}
        self._lock = threading.Lock()
        
    def update_keyboard_state(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                continue
                
            # Process frame using existing keyboard logic
            processed_frame = self.keyboard.process_frame(frame)
            
            with self._lock:
                self.current_text = self.keyboard.typed_text
                self.keyboard_state = {
                    "text": self.keyboard.typed_text,
                    "layout": self.keyboard.current_layout,
                    "shift_active": self.keyboard.shift_active,
                    "pressed_keys": list(self.keyboard.pressed_keys),
                    "current_keys": {
                        key: {"x": pos[0], "y": pos[1]} 
                        for key, pos in self.keyboard.current_keys.items()
                    }
                }

# Initialize keyboard server
keyboard_server = KeyboardServer()

# Start keyboard processing in background thread
processing_thread = threading.Thread(target=keyboard_server.update_keyboard_state)
processing_thread.daemon = True
processing_thread.start()

@app.route('/keyboard-state')
def get_keyboard_state():
    with keyboard_server._lock:
        return jsonify(keyboard_server.keyboard_state)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
