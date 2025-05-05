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

class ColourChaser(Node):
    def __init__(self):
        super().__init__('colour_chaser')
        self.bridge = CvBridge()

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.handle_image, 10)
        self.rate = self.create_rate(10)

        self.detect = {'red': False, 'green': False, 'blue': False}
        self.too_close = False

        cv2.namedWindow("view", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("view", 320, 240)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def move_forward(self, duration=1, speed=0.2):
        motion = Twist()
        motion.linear.x = speed
        for _ in range(duration * 10):
            self.cmd_pub.publish(motion)
            self.rate.sleep()
        self.stop_robot()

    def rotate(self, duration=1, angular=0.3):
        spin = Twist()
        spin.angular.z = angular
        for _ in range(duration * 10):
            self.cmd_pub.publish(spin)
            self.rate.sleep()
        self.stop_robot()

    def handle_image(self, data):
        image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        ranges = {
            'red': [
                (np.array([0, 120, 70]), np.array([10, 255, 255])),
                (np.array([170, 120, 70]), np.array([180, 255, 255]))
            ],
            'green': [(np.array([50, 100, 100]), np.array([70, 255, 255]))],
            'blue': [(np.array([110, 100, 100]), np.array([130, 255, 255]))]
        }

        for color, bounds in ranges.items():
            mask = None
            for lower, upper in bounds:
                current_mask = cv2.inRange(hsv, lower, upper)
                mask = current_mask if mask is None else cv2.bitwise_or(mask, current_mask)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                biggest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(biggest)

                if area > 1000:
                    self.detect[color] = True
                    print(f"{color.upper()} detected — area: {int(area)}")

                    rect = cv2.minAreaRect(biggest)
                    box = np.int0(cv2.boxPoints(rect))
                    center = tuple(map(int, rect[0]))

                    cv2.polylines(image, [box], True, (255, 255, 0), 2)
                    cv2.putText(image, color, (center[0]-40, center[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    if color == 'blue' and area > 300000:  # ~1 metre stop threshold
                        self.too_close = True
                else:
                    self.detect[color] = False

        cv2.imshow("view", image)
        cv2.waitKey(3)

def main():
    rclpy.init()
    node = ColourChaser()

    def shutdown(sig, frame):
        rclpy.shutdown()

    signal.signal(signal.SIGINT, shutdown)
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    pattern = [
        (1.5, 0),
        (0, 10),
        (1.2, 0),
        (0, 10),
        (1.6, 0),
        (0, 10),
        (1.6, 0),
        (0, 10)
    ]
    index = 0

    try:
        while rclpy.ok():
            if node.too_close:
                node.stop_robot()
                print("[✓] Stopped within ~1 metre of blue box.")
                break

            if node.detect['blue']:
                node.move_forward(1)
            elif index < len(pattern):
                turn, move = pattern[index]
                if turn > 0:
                    node.rotate(duration=int(turn))
                if move > 0:
                    node.move_forward(duration=int(move))
                index += 1
            else:
                node.rotate(duration=1)

    except ROSInterruptException:
        pass

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
