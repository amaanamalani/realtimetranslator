# realtimetranslator
Real Time Sign Language Translator to Speech. This is the capstone project I worked on in my final year of BTech (Data Science) degree. 
Dataset: Key Point data extracted using MediaPipe
Training data: 25 sign language actions, 20 sequences per action
Key Points: Face, Pose, Left Hand, Right Hand Key Points
Face Key Points = 468 x 3(x,y,z) = 1404 Key Points
Pose Key Points = 33 x 4(x,y,z, visibility) = 132 Key Points
Right Hand Key Points = 21 x 3(x,y,z) = 63 Key Points
Left Hand Key Points = 21 x 3(x,y,z) = 63 Key Points
Total Key Points: 1662 per frame
20 Sequences, 30 frames each - 500 frames in total
Sequences Flattening: Data sequences are flattened and mapped to labels for machine learning model training.
Labeling: Labels assigned based on the corresponding sign language gesture, facilitating supervised learning.

1. Data Structure: Organized into 25 action categories, each containing 20 sequences with 30 frames each.
2. Data Preprocessing: Flattened frames and mapped labels for training.
3. Machine Learning Models: Two models utilized - LSTM and Simple RNN.
4. Real-Time Translation: Application captures video from a webcam, processes it with MediaPipe and the trained model to generate real-time predictions.
5. Visualization: Recognized actions are displayed, and a sentence of recognized signs is maintained.
6. Speech Synthesis: pyttsx3 library used for text-to-speech synthesis.
7. User Interface: Streamlit employed to create an intuitive, user-friendly virtual environment.
