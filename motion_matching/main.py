import rclpy
from rclpy.node import Node

from motion_matching.node import MotionMatchingNode


def main(args=None):
    rclpy.init(args=args)

    node = Node('motion_matching_node')
    MotionMatchingNode(node)

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
