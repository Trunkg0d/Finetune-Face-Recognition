import cv2
import os
import time

# --- Configuration ---
# Number of images to capture in a burst
IMAGES_TO_CAPTURE = 15
# Key to trigger the capture
CAPTURE_KEY = 'c'
# Key to quit the program
QUIT_KEY = 'q'
# Directory to save the captured images
OUTPUT_DIR = "captures"

def main():
    """
    Main function to run the camera capture application.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # Initialize the camera. 0 is the default camera.
    # If you have multiple cameras, you can try changing this to 1, 2, etc.
    cap = cv2.VideoCapture(0)

    # Check if the camera was opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("\n--- Camera is active ---")
    print(f"Press '{CAPTURE_KEY}' to capture {IMAGES_TO_CAPTURE} photos.")
    print(f"Press '{QUIT_KEY}' to exit.")

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # If the frame was not read correctly, break the loop
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame in a window
        # Add text instructions on the video feed
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Press '{CAPTURE_KEY}' to Capture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Press '{QUIT_KEY}' to Quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Camera Feed', display_frame)

        # Wait for a key press (1 millisecond delay)
        key = cv2.waitKey(1) & 0xFF

        # If the capture key is pressed
        if key == ord(CAPTURE_KEY):
            print(f"\nCapture key '{CAPTURE_KEY}' pressed. Capturing {IMAGES_TO_CAPTURE} images...")
            
            # Loop to capture the specified number of images
            for i in range(IMAGES_TO_CAPTURE):
                # Read a fresh frame for each capture
                success, capture_frame = cap.read()
                if not success:
                    print(f"  Error: Failed to capture image {i+1}.")
                    continue

                # Generate a unique filename using a timestamp to avoid overwriting
                timestamp = int(time.time() * 1000)
                filename = f"{OUTPUT_DIR}/capture_{timestamp}_{i+1}.jpg"
                
                # Save the captured frame to a file
                cv2.imwrite(filename, capture_frame)
                print(f"  Saved: {filename}")
                
                # Optional: Add a small delay between captures if needed
                # time.sleep(0.1) 

            print("Capture complete. Returning to live view.")

        # If the quit key is pressed, break the loop
        elif key == ord(QUIT_KEY):
            print("Quit key pressed. Exiting...")
            break

    # When everything is done, release the camera and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()