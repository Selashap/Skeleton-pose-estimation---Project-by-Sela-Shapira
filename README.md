This is the github repository for the third assignment in the course advanced robotics at middlesex university.
In this repository I uploaded all the related files for this assignment.
This repository contains the following things: 

Demonstration Video.

Report.

Dataset.

Training model for Machine Learning.
                                            
Programs.

Demonstration Video:
The demonstration video is attached as a link to the YouTube video under the file named: "Demonstration Video"

Report:
The report is attached here in the repository as a PDF file named: "Human pose Estimation report - Sela Shapira M00956836"

Dateset:
I uploaded the dataset for training the model, the dataset called:"pictures_for_data"

Training model:
I aded to this repository the Machine Learning model called: "pose_recog_model.pkl"

Programs:
in this repository I uploaed 7 program files which are as follows:

1. assignment3.py - 

assignment3.py is the main program for the assignment, this program includes obstacle avoidance algorithm, and the turtlesim motion using machine Learning gestures prediction.
Moreover, in this code I added a target, using the go_to_goal function which is the target that the turtle is aiming to reach, with this code the turtle will move using my predicted gestures, avoid the obstacles (which are set as other turtles) and reach his final destination.

2. Data_collect_prog.py - 

Data_collect_prog.py is the program I wrote to collect the pictures for the Dataset, in this program I used the camera wih the opencv command, and I set a timer that during that time I stood in the gestures position, and the computer camera took pictures of me.

3. gesture_prediction.py - 

gesture_prediction.py is the program to predict the gestures, in this program I loaded the training model and used it to see if the model is predicting my gestures. forward, backward, left, right.

4. pose_recog_model.py - 

pose_recog_model.py is the program I wrote to train the machine learning model, in order to train the model I used the CNN method and loaded it into a training model called: "pose_recog_model.pkl"

5. Skeleton_detection.py - 

Skeleton_detection.py is the program I wrote to detect the skeleton of my body.

6. Skeleton_turtle_motion.py - 

Skeleton_turtle_motion.py is the program I wrote to make the turtle move according to my skeleton, in this program I used the camera vision, in which according to the gesture identified in the camera Twist messages for the turtle are published.

7. Skeleton_turtle_motion_withML.py - 

Skeleton_turtle_motion_withML.py is the program I wrote to make the turtle move according to the gestures prediction, in this code I loaded the Machine learning model, and using the predicted gesture, the appropriate Twist message is published and the turtle is moving according to that.
