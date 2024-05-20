import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import mediapipe as mp
import pickle
import pyttsx3
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

# Creating a Mediapipe Holistic Model
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Creating a Function to apply mediapipe detection on video captured by camera
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

# Creating a function of fancy landmarks 
def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                             mp_drawing.DrawingSpec(color=(27, 3, 87), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(84, 23, 235), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(27, 3, 87), thickness=2, circle_radius=2), 
                             mp_drawing.DrawingSpec(color=(84, 23, 235), thickness=2, circle_radius=1)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(27, 3, 87), thickness=2, circle_radius=2), 
                             mp_drawing.DrawingSpec(color=(84, 23, 235), thickness=2, circle_radius=1)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(27, 3, 87), thickness=2, circle_radius=2), 
                             mp_drawing.DrawingSpec(color=(84, 23, 235), thickness=2, circle_radius=1)
                             )

# Defining a Function to extract keypoints and flatten them or return zero array if not detected 
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'help', 'more', 'yes', 'phone', 'drink', 'eat', 'i', 'my', 'you', 'what', 'when', 'book', 'know', 'sentence', 'time', 'baby', 'happy', 'sad', 'look', 'fine', 'forget', 'go', 'like'])

# Loading the Pre-Trained Sequences and attached labels from pickled files
 
with open('sequences.pickle', 'rb') as file2:
    seq=pickle.load(file2)
    
with open('labels.pickle', 'rb') as file2:
    lab=pickle.load(file2)

# alloting value to X (sequences)
X = np.array(seq)

# alloting value to y (labels)
y = lab
# Encode labels into integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=7)

# Define the LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(128, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model_lstm.add(LSTM(64, activation='relu'))
model_lstm.add(Dense(actions.shape[0], activation='softmax'))  # Output layer with the number of classes

# Load the LSTM Model trained earlier
model_lstm.load_weights('lstm_model.h5')

# Compile the model
model_lstm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Predicting the result
res = model_lstm.predict(X_test)

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Function to convert text to speech
def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

# Set up Streamlit
st.title("Real-Time Sign Language Translator To Speech")

# Initialize state variables
cap = None
is_translating = False

start_translation_button = st.button("Start Translation", key="start_translation_button")

# Button to start translation
if start_translation_button:
    is_translating = True
    cap = cv2.VideoCapture(0)

stop_translation_button = st.button("Stop Translation", key="stop_translation_button")

# Detection variables
sequence = []
sentence = []
threshold = 0.8
previous_action = None

stframe = st.empty()

holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Set mediapipe model 
while is_translating:

    # Read feed
    ret, frame = cap.read()

    # Make detections
    image, results = mediapipe_detection(frame, holistic)
    # print(results)
        
    # Draw landmarks
    draw_styled_landmarks(image, results)
    
    # Prediction logic
    keypoints = extract_keypoints(results)
    # sequence.insert(0,keypoints)
    # sequence = sequence[:30]
    sequence.append(keypoints)
    sequence = sequence[-30:]
    
    if len(sequence) == 30:
        res = model_lstm.predict(np.expand_dims(sequence, axis=0))[0]
        predicted_action = actions[np.argmax(res)]
        print(predicted_action)

        # Check if the recognized action is different from the previous one
        if predicted_action != previous_action:
            text_to_speech(predicted_action)
            previous_action = predicted_action
            
    # Viz logic
        if res[np.argmax(res)] > threshold: 
            if len(sentence) > 0: 
                if actions[np.argmax(res)] != sentence[-1]:
                    sentence.append(actions[np.argmax(res)])
            else:
                sentence.append(actions[np.argmax(res)])

        if len(sentence) > 5: 
            sentence = sentence[-5:]
            
    cv2.rectangle(image, (0,0), (640, 40), (8, 8, 99), -1)
    cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
    # # Show to screen
    # cv2.imshow('Prediction Feed', image)

    # # Break gracefully
    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     break
    stframe.image(image, channels="BGR")

    # Button to stop translation
    if stop_translation_button:
        is_translating = False
        # cap.release()
        # cv2.destroyAllWindows()

if cap:
    cap.release()
cv2.destroyAllWindows()