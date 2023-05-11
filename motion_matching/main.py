from motion_matching.node import MotionMatchingNode

import argparse
import socketio
import rclpy
from rclpy.node import Node
import eventlet
from eventlet.green import threading
eventlet.monkey_patch()


sio = socketio.Server(cors_allowed_origins="*")
# the index.html file hosted by eventlet is a dummy file
# it appears to be required to host some html file..
app = socketio.WSGIApp(sio, static_files={
    '/': {'content_type': 'text/html', 'filename': 'index.html'}
})


def server():
    eventlet.wsgi.server(eventlet.listen(('', 5555)), app)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--timer', help='time for callback',
                        type=float, default=0.1)
    parser.add_argument('--camera_human', help='path for camera human',
                        type=str, default='/dev/video0')
    parser.add_argument('--camera_robot', help='path for camera robot',
                        type=str, default='/dev/video2')
    arg = parser.parse_args()

    threading.Thread(target=server).start()

    rclpy.init(args=args)

    node = Node('motion_matching_node')
    MotionMatchingNode(node, sio, arg.timer,
                       arg.camera_human, arg.camera_robot)

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
