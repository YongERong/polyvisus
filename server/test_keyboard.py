import requests
import time
import json

def test_keyboard_server():
    """
    Test the keyboard server by making requests and printing the responses
    """
    url = "http://localhost:5000/keyboard-state"
    
    print("Starting keyboard server test...")
    print("Press Ctrl+C to stop testing\n")
    
    try:
        while True:
            try:
                # Make request to server
                response = requests.get(url)
                data = response.json()
                
                # Clear terminal (optional)
                print("\033[H\033[J")  # Clear screen
                
                # Print current keyboard state
                print("=== Keyboard State ===")
                print(f"Current Text: {data['text']}")
                print(f"Current Layout: {data['layout']}")
                print(f"Shift Active: {data['shift_active']}")
                print("\nPressed Keys:", data['pressed_keys'])
                print("\nResponse Time: {:.3f}s".format(response.elapsed.total_seconds()))
                
                # Sleep briefly to avoid flooding the server
                time.sleep(0.1)
                
            except requests.exceptions.ConnectionError:
                print("Error: Could not connect to server. Is it running?")
                time.sleep(2)
                
            except json.JSONDecodeError:
                print("Error: Received invalid JSON from server")
                time.sleep(2)
                
    except KeyboardInterrupt:
        print("\nTest ended by user")

if __name__ == "__main__":
    test_keyboard_server()
