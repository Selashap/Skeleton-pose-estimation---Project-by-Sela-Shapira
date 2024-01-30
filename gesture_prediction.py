import keras
import cv2
import mediapipe as mp
from keras import layers
from tensorflow import data as tf_data
import tensorflow as tf
import joblib

vc = cv2.VideoCapture(0)

gesture_model = joblib.load('pose_recog_model.pkl')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cv2.namedWindow("gesture recognition", cv2.WINDOW_NORMAL)

while vc.isOpened():
    rval, frame = vc.read()  # read frame
    target_size = (120, 160)
    frame_resized = cv2.resize(frame, target_size)
    img_array = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) 
    results = pose.process(img_array)

    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame_resized, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
    img_array = img_array / 255.0  # Normalize
    img_array = tf.expand_dims(img_array, 0)

    predictions = gesture_model.predict(img_array)
    predicted_gesture_index = tf.argmax(predictions, axis=1)[0].numpy()
    gesture_names = ["backward", "forward", "left", "right"]
    predicted_gesture = gesture_names[predicted_gesture_index]
    
    cv2.resizeWindow("gesture recognition", target_size[1], target_size[0])
    
    cv2.imshow('gesture recognition', frame_resized)
    
    print(f"your gesture is: {predicted_gesture}")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vc.release()

cv2.destroyAllWindows()   
