import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model

model = load_model('action.h5')

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def extract_keypoints(results):
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(42)
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(42)
    return np.concatenate([lh, rh])


sequence = []
sentence = []
predictions = []
threshold = 0.8
actions = np.array(['hello', 'thanks', 'iloveyou', 'yes', 'no', 'please'])
cap = cv2.VideoCapture(0)

# --- NEW: Set a higher camera resolution ---
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# ------------------------------------------

frame_count = 0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame. Check if the camera is connected and available.")
            break

        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape
        frame_count += 1

        image, results = mediapipe_detection(frame, holistic)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-50:]

        if len(sequence) == 50 and frame_count % 5 == 0:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))

            if np.unique(predictions[-5:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

        if results.left_hand_landmarks or results.right_hand_landmarks:
            x_, y_ = [], []
            all_landmarks = []
            if results.left_hand_landmarks:
                all_landmarks.extend(results.left_hand_landmarks.landmark)
            if results.right_hand_landmarks:
                all_landmarks.extend(results.right_hand_landmarks.landmark)
            if all_landmarks:
                for landmark in all_landmarks:
                    x_.append(landmark.x)
                    y_.append(landmark.y)
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # --- ADDED: Display prediction above the bounding box ---
                if sentence:
                    cv2.putText(image, sentence[-1], (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

