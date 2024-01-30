import cv2
import mediapipe as mp
from geometry_msgs.msg import Twist 
import rclpy 
import joblib
from std_msgs.msg import String
from turtlesim.msg import Pose
import numpy as np

# Initializing ROS node
rclpy.init()
node = rclpy.create_node('turtle_gesture_control')

publisher = node.create_publisher(Twist, '/turtle1/cmd_vel', 10)
publisher_gesture = node.create_publisher(String, 'gesture_prediction', 10)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Global variables
obstacle_detected = False
obstacle_threshold = 0.5
goal_position = {'x': 8.0, 'y': 10.0}  # Setting the coordinates for the target

# Callback function for turtlesim pose
def pose_callback(msg):
	global obstacle_detected
	obstacle_detected = msg.x < obstacle_threshold or msg.y < obstacle_threshold

subscription = node.create_subscription(Pose, '/turtle1/pose', pose_callback, 10)

gesture_model = joblib.load('pose_recog_model.pkl')

cap = cv2.VideoCapture(0)

twist_msg = Twist()

while rclpy.ok():
	ret,frame = cap.read()
	if not ret:
		break 

	rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	
	results = pose.process(rgb_frame)
	
	linear_scale = 2.0
	angular_scale = 1.0
	
	 # Check for obstacle detection from the turtle's pose
	if obstacle_detected:
		# Get the relative position of the obstacle with respect to the turtle
		relative_position = "front"	

		if results.pose_landmarks:
			# Extracting landmark coordinates for specific body parts
			turtle_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x
	
			# Assuming that the obstacle is directly in front of the turtle
			if turtle_x < 0.5:
				relative_position = "right"
			else:
				relative_position = "left"
		
		# Adjust turning direction based on obstacle position
		if relative_position == "front":
			twist_msg.linear.x = 0.0
			twist_msg.angular.z = 1.0  # Turn right
		elif relative_position == "right":
			twist_msg.linear.x = 0.0
			twist_msg.angular.z = -1.0  # Turn left
		elif relative_position == "left":
			twist_msg.linear.x = 0.0
			twist_msg.angular.z = 1.0  # Turn right
	
	else:
		# If no obstacle is detected,continuing with the gestures recognition
		if results.pose_landmarks:
			# Extracting landmark coordinates for specific body parts
			left_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
			right_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

			left_hand_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
			right_hand_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
	
	        	# Thresholds for gesture recognition
			threshold_up = 0.5 
			threshold_down = 0.5
		
        		# Gesture recognition logic (for the turtle movement)
			forward_gesture = left_shoulder_y < threshold_up and right_shoulder_y < threshold_up
			backward_gesture = left_shoulder_y > threshold_down and right_shoulder_y > threshold_down
			right_gesture = left_shoulder_y < threshold_up and right_hand_y < threshold_up
			left_gesture = left_hand_y < threshold_up and right_shoulder_y < threshold_up
	
			twist_msg = Twist()
		
			twist_msg.linear.x = float(linear_scale) if forward_gesture else (-float(linear_scale) if backward_gesture else 0.0)
			twist_msg.angular.z = float(angular_scale) if right_gesture else (-float(angular_scale) if left_gesture else 0.0)
	
			# "Go to goal" behavior
			distance_to_goal = np.sqrt((results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x - goal_position['x'])**2 +(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y - goal_position['y'])**2)
	        	
			angle_to_goal = np.arctan2(goal_position['y'] - results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y,goal_position['x'] - results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x)
			
			if distance_to_goal < 0.5:
				twist_msg.linear.x = linear_scale
				twist_msg.angular.z = angular_scale * (angle_to_goal - np.arctan2(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y - goal_position['y'],
                                                             results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x - goal_position['x']))
	        	
	        	# Publishing the Twist message to control the turtle
			publisher.publish(twist_msg)
		
			resized_frame = cv2.resize(rgb_frame, (160, 120))
			resized_frame = resized_frame / 255.0 
		
			predictions = gesture_model.predict(resized_frame.reshape(1, 160, 120, 3))
			predicted_gesture_index = np.argmax(predictions)
			#predicted_gesture_index = tf.argmax(predictions, axis=1)[0].numpy()
			gesture_names = ["backward", "forward", "left", "right"]
			predicted_gesture = gesture_names[predicted_gesture_index]
			publisher_gesture.publish(String(data=predicted_gesture))
		
			mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        	
	cv2.imshow('Pose Skeleton and turtle control', frame)
        
	if cv2.waitKey(1) & 0xFF == ord('q'):
        	break
        	
cap.release()
cv2.destroyAllWindows()

rclpy.shutdown()
