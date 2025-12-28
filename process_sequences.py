import cv2
import numpy as np
import os
import mediapipe as mp

# --- Setup MediaPipe ---
mp_holistic = mp.solutions.holistic  # A model that includes hand tracking
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapipe_detection(image, model):
    """Processes an image and returns the MediaPipe results."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def extract_keypoints(results):
    """
    Extracts and normalizes hand keypoints from MediaPipe results.
    Normalization makes the model robust to hand position and scale.
    """
    # Create empty arrays for left and right hand landmarks
    lh = np.zeros(21 * 2)
    rh = np.zeros(21 * 2)

    # --- Extract Left Hand landmarks ---
    if results.left_hand_landmarks:
        landmarks = results.left_hand_landmarks.landmark
        # Get all x and y coordinates
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        # Find min values for normalization
        min_x, min_y = min(x_coords), min(y_coords)
        # Normalize and flatten
        lh = np.array([[lm.x - min_x, lm.y - min_y] for lm in landmarks]).flatten()

    # --- Extract Right Hand landmarks ---
    if results.right_hand_landmarks:
        landmarks = results.right_hand_landmarks.landmark
        # Get all x and y coordinates
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        # Find min values for normalization
        min_x, min_y = min(x_coords), min(y_coords)
        # Normalize and flatten
        rh = np.array([[lm.x - min_x, lm.y - min_y] for lm in landmarks]).flatten()

    return np.concatenate([lh, rh])


# --- Define Paths and Variables ---
# IMPORTANT: These must match the values in your collect_sequences.py script
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'thanks', 'iloveyou', 'yes', 'no', 'please'])
num_sequences = 50
sequence_length = 50

# --- Main Data Processing Loop ---
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Loop through actions, then sequences
    for action in actions:
        for sequence in range(num_sequences):
            window = []  # This will hold the 30 frames of landmark data for one video
            # Loop through each frame in the sequence
            for frame_num in range(sequence_length):
                frame_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.jpg")
                frame = cv2.imread(frame_path)

                # Add a check to ensure the frame was loaded correctly
                if frame is None:
                    print(f"Warning: Could not read frame {frame_path}. Skipping.")
                    # Append a zero-array if frame is missing to maintain sequence length
                    window.append(np.zeros(21 * 2 * 2))  # 84 zeros
                    continue

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Extract keypoints
                keypoints = extract_keypoints(results)
                window.append(keypoints)

            # --- FIX: Save the processed sequence as a single .npy file ---
            # The file is named after the sequence number (e.g., 0.npy, 1.npy, etc.)
            npy_path = os.path.join(DATA_PATH, action, f"{sequence}.npy")
            np.save(npy_path, np.array(window))

            print(f"Processed and saved: {action}, sequence {sequence}")

print("--- Data processing complete! ---")