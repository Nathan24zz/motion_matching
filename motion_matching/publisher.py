import rclpy
from std_msgs.msg import String
from tachimawari_interfaces.msg import SetJoints
from tachimawari_interfaces.msg import Joint
from typing import List


class MotionMatchingPublisher:
    def __init__(self, node: rclpy.Node):
        self.node = node

        self.joints: List[Joint] = []
        self.publisher = self.node.create_publisher(
            SetJoints, 'joint/set_joints', 10)

        timer_period = 0.5
        self.timer = self.node.create_timer(timer_period, self.timer_callback)

        self.init_joints()

    def timer_callback(self):
        self.node.get_logger().info('Counting')

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
