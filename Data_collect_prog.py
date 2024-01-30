import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# Set the gesture label for the current recording
gestures = ["forward", "backward", "left", "right"]

# Counter for frame numbering
frame_count = 0

# Duration (in seconds) to capture frames for each gesture
gesture_duration = 40

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

for current_gesture in gestures:
    print(f"Capture frames for {current_gesture} gesture. Get ready!")

    # Capture frames for the specified duration
    start_time = time.time()
    while (time.time() - start_time) < gesture_duration:
        # Capture frame-by-frame
        ret, frame = cap.read()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb_frame)

        # Display the resulting frame with skeleton overlay
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Capture Frames", frame)

        # Save frames with the current gesture label
        frame_count += 1
        file_name = f"pictures_for_data/{current_gesture}/{frame_count}.jpg"
        cv2.imwrite(file_name, frame)
        print(f"Saved: {file_name}")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"{gesture_duration} seconds for {current_gesture} completed.")
    
    time.sleep(5)

cap.release()

cv2.destroyAllWindows()

