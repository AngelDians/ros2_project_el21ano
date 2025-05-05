import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time


class ColourDetector(Node):
    def __init__(self):
        super().__init__('colour_detector')
        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.goal_sent = False

        # HSV bounds
        self.bounds = {
            'red': [
                (np.array([0, 120, 70]), np.array([10, 255, 255])),
                (np.array([170, 120, 70]), np.array([180, 255, 255]))
            ],
            'green': [
                (np.array([35, 100, 100]), np.array([85, 255, 255]))
            ],
            'blue': [
                (np.array([90, 100, 100]), np.array([130, 255, 255]))
            ]
        }

        # Start Nav2 after delay
        self.get_logger().info('Waiting for Nav2...')
        time.sleep(10)
        self.get_logger().info('Sending goal...')
        self.send_goal(2.0, 2.0, 0.0)  # You can tweak this

    def send_goal(self, x, y, theta):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = np.sin(theta / 2.0)
        goal_msg.pose.pose.orientation.w = np.cos(theta / 2.0)

        self.nav_client.wait_for_server()
        self.nav_client.send_goal_async(goal_msg)

    def image_callback(self, msg):
        if self.goal_sent:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        found_blue_close = False

        for colour, ranges in self.bounds.items():
            mask = None
            for lower, upper in ranges:
                m = cv2.inRange(hsv, lower, upper)
                mask = m if mask is None else cv2.bitwise_or(mask, m)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)

                if area > 1000:
                    print(f"{area:.1f}  {colour} found")

                    if colour == "blue" and area > 3000:
                        found_blue_close = True

        if found_blue_close:
            self.get_logger().info("Blue object detected within 1m. Cancelling goal.")
            self.cancel_goal()
            self.goal_sent = True

    def cancel_goal(self):
        self.nav_client._goal_handle.cancel_goal_async()


def main():
    rclpy.init()
    node = ColourDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
