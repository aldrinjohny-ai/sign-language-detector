import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# --- Define Paths and Variables (Must match your data processing script) ---
DATA_PATH = os.path.join('MP_Data')
# FIX: Updated actions to include all six words
actions = np.array(['hello', 'thanks', 'iloveyou', 'yes', 'no', 'please'])
# Videos are going to be 30 frames in length
sequence_length = 50
# Thirty videos worth of data
num_sequences = 50

# --- Step 1: Load Data and Create Labels ---
sequences, labels = [], []
label_map = {label: num for num, label in enumerate(actions)}

print("Loading data...")
for action in actions:
    for sequence in range(num_sequences):
        # Construct the full path to the .npy file
        file_path = os.path.join(DATA_PATH, action, str(sequence) + '.npy')

        # Check if the file exists before trying to load it
        if os.path.exists(file_path):
            res = np.load(file_path)
            sequences.append(res)
            labels.append(label_map[action])
        else:
            print(f"Warning: File not found {file_path}. Skipping.")

print(f"Data loaded for {len(sequences)} sequences.")

# Convert lists to numpy arrays
X = np.array(sequences)
# One-hot encode the labels
y = to_categorical(labels).astype(int)

# --- Step 2: Split Data into Training and Testing Sets ---
# FIX: Increased test_size for a more robust evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 3: Build and Compile the LSTM Model ---
model = Sequential()
# Input layer: 30 frames, 84 landmarks per frame (21 for each hand, with x and y)
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 84)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
# This layer does not return sequences, as it's the last LSTM layer
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
# Output layer: uses softmax for multi-class classification
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Print a summary of the model architecture
model.summary()

# --- Step 4: Train the Model ---
print("\nTraining model...")
# The model will train for 200 epochs. You can adjust this value.
model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test))
print("Training complete.")

# --- Step 5: Save the Model ---
# The trained model is saved to a file, so we can use it for real-time detection later
model.save('action.h5')
print("\nModel trained and saved as action.h5")
