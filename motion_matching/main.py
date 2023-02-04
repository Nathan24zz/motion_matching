import rclpy
from rclpy.node import Node

from motion_matching.publisher import MotionMatchingPublisher


def main(args=None):
    rclpy.init(args=args)

    node = Node('motion_matching_node')

    minimal_publisher = MotionMatchingPublisher(node)

    rclpy.spin(minimal_publisher.node)

    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
