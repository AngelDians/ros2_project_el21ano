import threading
import sys, time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
import signal
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from math import sin, cos


class ColourExplorer(Node):
    def __init__(self):
        super().__init__('colour_explorer')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_subscriber = self.create_subscription(Image, '/camera/image_raw', self.process_image, 10)
        self.bridge = CvBridge()
        self.rate = self.create_rate(10)

        # Detection flags
        self.seen = {'red': False, 'green': False, 'blue': False}
        self.too_close = False

        cv2.namedWindow('camera_feed', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('camera_feed', 320, 240)

    def move_forward(self, duration_sec=2, speed=0.2):
        msg = Twist()
        msg.linear.x = speed
        for _ in range(duration_sec * 10):
            self.publisher.publish(msg)
            self.rate.sleep()
        self.stop()

    def rotate(self, angle_rads=1.57, angular_speed=0.3):
        twist = Twist()
        duration = int(abs(angle_rads / angular_speed) * 10)
        twist.angular.z = angular_speed if angle_rads > 0 else -angular_speed
        for _ in range(duration):
            self.publisher.publish(twist)
            self.rate.sleep()
        self.stop()

    def stop(self):
        self.publisher.publish(Twist())

    def process_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        bounds = {
            'red': [
                (np.array([0, 120, 70]), np.array([10, 255, 255])),
                (np.array([170, 120, 70]), np.array([180, 255, 255]))
            ],
            'green': [(np.array([50, 100, 100]), np.array([70, 255, 255]))],
            'blue': [(np.array([110, 100, 100]), np.array([130, 255, 255]))]
        }

        for colour, ranges in bounds.items():
            mask = None
            for lower, upper in ranges:
                m = cv2.inRange(hsv, lower, upper)
                mask = m if mask is None else cv2.bitwise_or(mask, m)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                if area > 100:
                    self.seen[colour] = True
                    print(f"[{colour.upper()}] Found — Area: {area:.1f}")

                    rect = cv2.minAreaRect(largest)
                    box = np.int0(cv2.boxPoints(rect))
                    center = tuple(map(int, rect[0]))

                    cv2.polylines(frame, [box], True, (255, 255, 0), 2)
                    cv2.putText(frame, f"{colour.capitalize()}", (center[0]-30, center[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    if colour == 'blue' and area > 180000:
                        self.too_close = True
                else:
                    self.seen[colour] = False

        cv2.imshow('camera_feed', frame)
        cv2.waitKey(3)

def main():
    rclpy.init()
    node = ColourExplorer()

    def shutdown_handler(sig, frame):
        rclpy.shutdown()
    signal.signal(signal.SIGINT, shutdown_handler)

    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    try:
        path = [
            (1.57, 0),
            (0, 10),
            (1.2, 0),
            (0, 10),
            (1.6, 0),
            (0, 10),
            (1.6, 0),
            (0, 10)
        ]
        path_index = 0

        while rclpy.ok():
            if node.too_close:
                node.stop()
                print("[TASK COMPLETE] Blue box reached within 1m.")
                break

            if node.seen['blue'] and not node.too_close:
                print("[ACTION] Moving toward blue...")
                node.move_forward(2)

            elif any(node.seen.values()):
                # Seen red/green but not blue → explore further
                print("[ACTION] Exploring more of map...")
                node.rotate(1.0)
                node.move_forward(2)

            elif path_index < len(path):
                angle, forward_time = path[path_index]
                if angle != 0:
                    node.rotate(angle)
                if forward_time != 0:
                    node.move_forward(forward_time)
                path_index += 1

            else:
                print("[IDLE] Nothing detected — rotating to search...")
                node.rotate(0.7)

    except ROSInterruptException:
        pass

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
