import cv2
import math
import mediapipe as mp
from rclpy.node import Node
from tachimawari_interfaces.msg import SetJoints
from tachimawari_interfaces.msg import Joint
from typing import List


class MotionMatchingNode:
    def __init__(self, node: Node):
        self.node = node

        self.joints: List[Joint] = []
        self.publisher = self.node.create_publisher(
            SetJoints, 'joint/set_joints', 10)

        timer_period = 0.5
        self.timer = self.node.create_timer(timer_period, self.timer_callback)

        self.init_joints()

    def timer_callback(self):
        self.node.get_logger().info('Counting')

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose

        # For webcam input:
        cap = cv2.VideoCapture(0)
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_height, image_width, _ = image.shape
                results = pose.process(image)
                
                if not results.pose_landmarks:
                    continue

                left_shoulder_x = results.pose_landmarks.landmark[11].x * image_width
                left_shoulder_y = results.pose_landmarks.landmark[11].y * image_height
                right_shoulder_x = results.pose_landmarks.landmark[12].x * image_width
                right_shoulder_y = results.pose_landmarks.landmark[12].y * image_height
                right_elbow_x = results.pose_landmarks.landmark[14].x * image_width
                right_elbow_y = results.pose_landmarks.landmark[14].y * image_height
                # sudut = math.acos((right_elbow_x * right_shoulder_x + right_elbow_y * right_shoulder_y) / (math.sqrt(right_elbow_x**2 + right_elbow_y**2) * math.sqrt(right_shoulder_x**2 + right_shoulder_y**2)))
                sudut = math.atan((right_elbow_x - right_shoulder_x) / (right_elbow_y - right_shoulder_y))

                print('left_shoulder_x: ', left_shoulder_x)
                print('left_shoulder_y: ', left_shoulder_y)
                print('right_shoulder_x: ', right_shoulder_x)
                print('right_shoulder_y:', right_shoulder_y)
                print('right_elbow_x: ', right_elbow_x)
                print('right_elbow_y:', right_elbow_y)
                print('====================')
                print('sudut: ', sudut*180/math.pi)
                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
                # cv2.imshow('MediaPipe Pose', image)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()


    def init_joints(self):
        for i in range(3, 7):
            joint = Joint()
            joint.id = i
            joint.position = 0.0

            self.joints.append(joint)

    def set_joint(self, id, position):
        for joint in self.joints:
            if joint.id == id:
                joint.position = position
