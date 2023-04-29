import base64
import cv2
import math
import mediapipe as mp
from rclpy.node import Node, MsgType
from tachimawari_interfaces.msg import SetJoints, Joint, CurrentJoints
from typing import List

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


class MotionMatchingNode:
    def __init__(self, node: Node, sio):
        self.node = node
        self.first = True
        self.sio = sio
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.joints: List[Joint] = []
        self.current_joints: List[Joint] = []
        self.publisher = self.node.create_publisher(
            SetJoints, 'joint/set_joints', 10)
        self.joint_subcriber = self.node.create_subscription(
            CurrentJoints, '/joint/current_joints', self.listener_callback, 10)

        timer_period = 0.1
        self.timer = self.node.create_timer(timer_period, self.timer_callback)
        self.save_motion = self.node.create_timer(0.5, self.timer_save_motion)

        self.image_height = 0
        self.image_width = 0
        self.results = 0

        self.speed = 0.1
        self.right_angle = 0.0
        self.left_angle = 0.0
        self.bottom_right_angle = 0.0
        self.bottom_left_angle = 0.0

        self.state_recording = False

        self.init_joints()

        def state_recording(sid, data):
            self.state_recording = data
            print('--state_recording-- ', self.state_recording)
        self.sio.on('state_recording', handler=state_recording)

    def calculate_angle(self, i1, i2, inverse=False):
        x1 = self.results.pose_landmarks.landmark[i1].x * self.image_width
        y1 = self.results.pose_landmarks.landmark[i1].y * self.image_height
        x2 = self.results.pose_landmarks.landmark[i2].x * self.image_width
        y2 = self.results.pose_landmarks.landmark[i2].y * self.image_height

        angle = (math.atan(
            (x2 - x1) / (y2 - y1))) * 180 / math.pi

        if (angle < 0):
            angle *= -1
        else:
            angle = 180 - angle

        angle = angle if not inverse else (180 - angle)

        if (x2 > x1 and not inverse) or (x2 < x1 and inverse):
            angle += 180

        return angle

    def clamp_value(self, val, bottom, top):
        return min(top, max(val, bottom))

    def get_reduced_value(self, current, target):
        return current + ((target - current) * self.speed)

    def listener_callback(self, message: MsgType):
        if (message.joints != []):
            self.current_joints = message

    def timer_save_motion(self):
        self.node.get_logger().info('-----')
        # record join robot and save to json

    def timer_callback(self):
        self.node.get_logger().info('Counting')
        self.sio.sleep(0.01)

        if self.first:
            # For webcam input:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.first = False

        success, image = self.cap.read()

        if not success:
            print("Ignoring empty camera frame.")
        else:
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image_height, self.image_width, _ = image.shape
            self.results = self.pose.process(image)
            # Draw the pose annotation on the image
            mp_drawing.draw_landmarks(
                image,
                self.results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            image = cv2.resize(image, (320, 240))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # encode and send to client
            # from image to binary buffer
            _, image = cv2.imencode('.jpg', image)
            # convert to base64 format
            data = base64.b64encode(image)
            self.sio.emit('image', data)

            if self.results.pose_landmarks:
                body_angle = 90 - self.calculate_angle(11, 12)
                right_angle = self.calculate_angle(12, 14) + body_angle
                bottom_right_angle = self.calculate_angle(
                    14, 16) - right_angle
                left_angle = self.calculate_angle(11, 13, True) - body_angle
                bottom_left_angle = self.calculate_angle(
                    13, 15, True) - left_angle

                # adjust to real robot's state
                bottom_right_angle *= -1
                left_angle *= -1

                # adjust to real robot's offset
                right_angle -= 45
                left_angle += 45

                right_angle = self.clamp_value(right_angle, -30, 100)
                left_angle = self.clamp_value(left_angle, -100, 30)
                bottom_right_angle = self.clamp_value(
                    bottom_right_angle, -120, 10)
                bottom_left_angle = self.clamp_value(
                    bottom_left_angle, -10, 120)

                self.right_angle = self.get_reduced_value(
                    self.right_angle, right_angle)
                self.left_angle = self.get_reduced_value(
                    self.left_angle, left_angle)
                self.bottom_right_angle = self.get_reduced_value(
                    self.bottom_right_angle, bottom_right_angle)
                self.bottom_left_angle = self.get_reduced_value(
                    self.bottom_left_angle, bottom_left_angle)

                self.set_joint(3, self.right_angle)
                self.set_joint(4, self.left_angle)
                self.set_joint(5, self.bottom_right_angle)
                self.set_joint(6, self.bottom_left_angle)

                set_joints = SetJoints()
                set_joints.joints = self.joints
                self.publisher.publish(set_joints)

                print('=================================================')
                print('body_angle', body_angle)
                print('right_angle', right_angle)
                print('bottom_right_angle', bottom_right_angle)
                print('left_angle', left_angle)
                print('bottom_left_angle', bottom_left_angle)

        # cap.release()

    def init_joints(self):
        for i in range(3, 7):
            joint = Joint()
            joint.id = i
            joint.position = 0.0

            self.joints.append(joint)

    def set_joint(self, id, position):
        for joint in self.joints:
            if joint.id == id:
                joint.position = float(position)
