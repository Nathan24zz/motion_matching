import base64
import cv2
import json
import math
import mediapipe as mp
from motion_matching.utils.json_utils import joint_id_by_name
from motion_matching.utils.file_utils import get_absolute_file_paths
from motion_matching.utils.score import Score
from motion_matching.utils.image_comparison import load_robot_rcnn_model, human_mediapipe_detection, robot_rcnn_detection
import numpy as np
import os
import time
from typing import List
from tqdm import tqdm
from fp_interfaces.msg import StateAndButton
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from rclpy.node import Node, MsgType
from akushon_interfaces.msg import RunAction
from tachimawari_interfaces.msg import SetJoints, Joint, CurrentJoints

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


class MotionMatchingNode:
    def __init__(self, node: Node, camera_human_path: str, camera_robot_path: str):
        self.node = node
        self.run_motion = False
        self.done_process_pose_comparison = False
        self.send_image_to_web = True
        self.first_time_camera_human = True
        self.first_time_camera_robot = True
        # self.sio = sio

        # Model for human and robot
        self.robot_model = None
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.joints: List[Joint] = []
        self.current_joints: List[Joint] = []
        self.joint_publisher = self.node.create_publisher(
            SetJoints, 'joint/set_joints', 10)
        self.joint_subcriber = self.node.create_subscription(
            CurrentJoints, '/joint/current_joints', self.listener_callback, 10)
        
        # interaction with server
        self.state_subscriber = self.node.create_subscription(
            StateAndButton, 'state_and_button', self.state_listener_callback, 10)
        self.human_image_publisher = self.node.create_publisher(
            Image, 'human_image', 10)
        self.robot_image_publisher = self.node.create_publisher(
            Image, 'robot_image', 10)
        self.br = CvBridge()

        # akushon
        self.run_action_pub = self.node.create_publisher(
            RunAction, 'action/run_action', 10)

        self.timer = self.node.create_timer(0.1, self.timer_callback)
        self.save_motion = self.node.create_timer(0.5, self.timer_save_motion)

        self.image_height = 0
        self.image_width = 0
        self.results = 0

        self.speed = 0.1
        self.right_angle = 0.0
        self.left_angle = 0.0
        self.bottom_right_angle = 0.0
        self.bottom_left_angle = 0.0

        # STATE in website
        self.state = ""
        self.state_recording = ""

        # JSON config for aruku
        self.json_dict = {
            "name": "Motion Final Project",
            "next": "motion_fp",
            "poses": [],
            "start_delay": 0,
            "stop_delay": 0
        }
        self.count = 0

        self.camera_human_path = camera_human_path
        self.camera_robot_path = camera_robot_path
        self.count_video = 0

        self.init_joints()

        # # Handler client data
        # def state(sid, data):
        #     if self.state != data:
        #         self.reinit()
        #     self.state = data

        # def state_recording(sid, data):
        #     if self.state_recording != data:
        #         self.reinit()

        #     self.state_recording = data
        #     if self.state == "play" and self.state_recording == "start":
        #         self.send_image_to_web = False

        # self.sio.on('state_recording', handler=state_recording)
        # self.sio.on('state', handler=state)

    def reinit(self):
        self.run_motion = False
        self.done_process_pose_comparison = False

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
    
    def state_listener_callback(self, message: MsgType):
        if self.state != message.state or  self.state_recording != message.button:
            self.reinit()
        self.state = message.state
        self.state_recording = message.button

        if self.state == "play" and self.state_recording == "start":
            self.send_image_to_web = False

    def listener_callback(self, message: MsgType):
        if (message.joints != []):
            for joint in message.joints:
                self.current_joints.append(joint)

    def timer_save_motion(self):
        self.node.get_logger().info('Save Motion')
        print('--STATE-- ', self.state)
        print('--state_recording-- ', self.state_recording)

        # record join robot and save to json
        if self.state == 'recording' and len(self.current_joints):
            if self.state_recording == "start":
                print('------RECORD----')
                joints = {}
                name = f"motion_{self.count}"
                pause = 0
                # TODO: need to adjust speed when trying on real robot
                speed = 0.05
                for joint in self.current_joints:
                    joints[joint_id_by_name[joint.id]] = joint.position

                poses_dict = {}
                poses_dict["joints"] = joints
                poses_dict["name"] = name
                poses_dict["pause"] = pause
                poses_dict["speed"] = speed
                self.json_dict["poses"].append(poses_dict)
                self.count += 1
            elif self.state_recording == "stop":
                print('------SAVE----')
                json_object = json.dumps(self.json_dict, indent=4)
                # print('json_object;', json_object)
                with open("data/json/motion_fp.json", "w") as outfile:
                    outfile.write(json_object)

    def timer_callback(self):
        while True:
            start_time = time.time()
            self.node.get_logger().info('Counting')
            # self.sio.sleep(0.01)

            # COMPARE POSE HUMAN AND ROBOT
            if self.state == "play" and self.state_recording == "stop" \
                    and not self.done_process_pose_comparison:
                human_dir = get_absolute_file_paths('data/image/human/')
                human_dir = sorted(human_dir, key=lambda t: os.stat(t).st_mtime)
                robot_dir = get_absolute_file_paths('data/image/robot/')
                robot_dir = sorted(robot_dir, key=lambda t: os.stat(t).st_mtime)
                result = cv2.VideoWriter('data/result.avi', 
                            cv2.VideoWriter_fourcc(*'DIVX'),
                            2, (640, 480))

                s = Score()
                image_1_points = []
                image_2_points = []

                if self.robot_model == None:
                    print('init robot model')
                    self.robot_model = load_robot_rcnn_model(
                        'weight/keypoint_rcnn.xml')
                # reinit mediapipe
                self.pose = mp_pose.Pose(
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
                if len(human_dir) > 0 and len(robot_dir) > 0:
                    if len(human_dir) <= len(robot_dir):
                        iter = len(human_dir)
                    else:
                        iter = len(robot_dir)

                    total_score = 0
                    count = 0
                    for i in tqdm(range(iter), desc='Pose Comparison Process'):
                        # try:
                        img = np.zeros([480,640,3])
                        image_1_points, image1, imagestick1 = human_mediapipe_detection(
                            human_dir[i], self.pose)
                        image_2_points, image2, imagestick2 = robot_rcnn_detection(
                            robot_dir[i], self.robot_model)
                        
                        # arrange image in video
                        img[:240, :320] = image1
                        img[:240, 320:] = image2
                        img[240:, :320] = imagestick1
                        img[240:, 320:] = imagestick2
                        img = np.uint8(img)

                        final_score, score_list = s.compare(np.asarray(
                            image_1_points), np.asarray(image_2_points))
                        # print("Total Score : ", final_score)
                        # print("Score List : ", score_list)
                        total_score += final_score
                        count += 1

                        # add text in video
                        img = cv2.putText(img, "Score: " + str(round(final_score, 2)), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
                        img = cv2.putText(img, "Overall Score: " + str(round(total_score/count, 2)), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
                        result.write(img)
                    print('Overall Score: ', total_score/count)
                    result.release()
                self.done_process_pose_comparison = True

            # ----------------------CAMERA AND IMAGE----------------------
            # OPEN SECOND CAMERA WHEN STATE PLAY
            if self.state == "play":
                if self.first_time_camera_robot:
                    # For webcam input
                    self.cap_robot = cv2.VideoCapture(self.camera_robot_path)
                    self.cap_robot.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap_robot.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.first_time_camera_robot = False

                success, image = self.cap_robot.read()
                if not success:
                    print("Ignoring empty camera frame robot.")
                else:
                    if self.state_recording == "start":
                        print('save video robot')
                        cv2.imwrite(f'data/image/robot/img_{self.count_video}.jpg', image)
                    image = cv2.resize(image, (320, 240))

                    # _, image = cv2.imencode('.jpg', image)
                    # data = base64.b64encode(image)

                    if self.send_image_to_web:
                        robot_frame = self.br.cv2_to_imgmsg(image)
                        self.robot_image_publisher.publish(robot_frame)
                        # self.sio.emit('robot_image', data)

            # OPEN FIRST CAMERA WHEN STATE PLAY AND RECORDING
            if self.first_time_camera_human:
                # For webcam input:
                self.cap_human = cv2.VideoCapture(self.camera_human_path)
                self.cap_human.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap_human.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.first_time_camera_human = False

            success, image = self.cap_human.read()
            # perform mediapipe pose on image
            if not success:
                print("Ignoring empty camera frame human.")
            else:
                if self.state == "play" and self.state_recording == "start":
                    print('save video human')
                    cv2.imwrite(
                        f'data/image/human/img_{self.count_video}.jpg', image)
                    self.count_video += 1
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

                # # encode and send to client
                # # from image to binary buffer
                # _, image = cv2.imencode('.jpg', image)
                # # convert to base64 format
                # data = base64.b64encode(image)

                # alleviate delay
                if self.send_image_to_web:
                    human_frame = self.br.cv2_to_imgmsg(image)
                    self.human_image_publisher.publish(human_frame)
                    # self.sio.emit('human_image', data)

            # ----------------------ROBOT BEHAVIOUR----------------------
            if self.state == "recording":
                # robot mimic human's move when button is started
                if self.state_recording == "start" and self.results.pose_landmarks:
                    body_angle = 90 - self.calculate_angle(11, 12)
                    right_angle = self.calculate_angle(12, 14) + body_angle
                    bottom_right_angle = self.calculate_angle(
                        14, 16) - right_angle
                    left_angle = self.calculate_angle(11, 13, True) - body_angle
                    bottom_left_angle = self.calculate_angle(
                        13, 15, True) - left_angle

                    # TODO: need to adjust on real robot
                    # adjust to real robot's state
                    bottom_left_angle *= -1
                    left_angle *= -1

                    # adjust to real robot's offset
                    right_angle -= 45
                    left_angle += 45

                    right_angle = self.clamp_value(right_angle, -30, 90)
                    left_angle = self.clamp_value(left_angle, -30, 90)
                    # TODO: need to adjust on real robot
                    bottom_right_angle = self.clamp_value(
                        bottom_right_angle, -10, 120)
                    bottom_left_angle = self.clamp_value(
                        bottom_left_angle, -120, 10)

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
                    self.joint_publisher.publish(set_joints)

                    print('=================================================')
                    print('body_angle', body_angle)
                    print('right_angle', right_angle)
                    print('bottom_right_angle', bottom_right_angle)
                    print('left_angle', left_angle)
                    print('bottom_left_angle', bottom_left_angle)
            elif self.state == "play" and self.state_recording == "start":
                # run akushon based on json file earlier
                if not self.run_motion:
                    print("------RUN PUBLISHER--------")
                    with open("data/json/motion_fp.json") as file:
                        data = json.load(file)
                        data = json.dumps(data)

                        run_action_msg = RunAction()
                        run_action_msg.control_type = 1
                        run_action_msg.action_name = "Motion Final Project"
                        run_action_msg.json = data
                        self.run_action_pub.publish(run_action_msg)
                        self.run_motion = True
            print("--- %s seconds ---" % (time.time() - start_time))

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
