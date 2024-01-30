import cv2
import mediapipe as mp
from geometry_msgs.msg import Twist 
import rclpy 

# Initializing ROS node
rclpy.init()
node = rclpy.create_node('turtle_gesture_control')

publisher = node.create_publisher(Twist, '/turtle1/cmd_vel', 10)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

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
	
	        # Publishing the Twist message to control the turtle
		publisher.publish(twist_msg)
	
		mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        	
	cv2.imshow('Pose Skeleton and turtle control', frame)
        
	if cv2.waitKey(1) & 0xFF == ord('q'):
        	break
        	
cap.release()
cv2.destroyAllWindows()

rclpy.shutdown()
