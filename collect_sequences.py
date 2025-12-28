import cv2
import numpy as np
import os

# Create a new folder to store the sequence data
DATA_PATH = os.path.join('MP_Data')

# Define the dynamic signs (actions) you want to collect
# You can change these to 'yes', 'no', 'thankyou', etc.
actions = np.array(['hello', 'thanks', 'iloveyou','yes','no','please'])

# Define the number of videos (sequences) to collect for each action
num_sequences = 100

# Define the length of each video in frames
sequence_length = 100

# --- Folder Setup ---
for action in actions:
    for sequence in range(num_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# --- Data Collection ---
cap = cv2.VideoCapture(0)

# Loop through each action
for action in actions:
    # Loop through each sequence (video)
    for sequence in range(num_sequences):

        # --- Wait for user input to start recording ---
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f'Ready for "{action}", video {sequence}? Press "S" to Start.', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', frame)

            # Wait for 's' key to be pressed
            if cv2.waitKey(10) & 0xFF == ord('s'):
                break

        # --- Record the sequence ---
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            # Display recording status
            cv2.putText(frame, f'RECORDING: "{action}", video {sequence}, frame {frame_num}', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', frame)

            # Save the captured frame as an image file
            frame_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            cv2.imwrite(f"{frame_path}.jpg", frame)

            # Allow quitting with 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

cap.release()
cv2.destroyAllWindows()