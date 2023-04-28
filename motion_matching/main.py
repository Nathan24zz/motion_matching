import rclpy
from rclpy.node import Node
import eventlet
from eventlet.green import threading
eventlet.monkey_patch()
import socketio

from motion_matching.node import MotionMatchingNode

sio = socketio.Server(cors_allowed_origins="*")
# the index.html file hosted by eventlet is a dummy file
# it appears to be required to host some html file.. 
app = socketio.WSGIApp(sio, static_files={
    '/': {'content_type': 'text/html', 'filename': 'index.html'}
})

def server():
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)

def main(args=None):
    threading.Thread(target=server).start()

    rclpy.init(args=args)

    node = Node('motion_matching_node')
    MotionMatchingNode(node, sio)

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
