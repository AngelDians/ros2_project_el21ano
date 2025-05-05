import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
from math import sin, cos
import time

class CourseworkBot(Node):
    def __init__(self):
        super().__init__('coursework_bot')

        self.bridge = CvBridge()
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

        self.seen_blue = False
        self.close_to_blue = False
        self.goal_handle = None
        self.goals = [
            (2.0, 2.0, 0.0),
            (2.0, -2.0, 0.0),
            (-2.0, -2.0, 1.57),
            (-2.0, 2.0, 3.14),
            (0.0, 0.0, 0.0)
        ]
        self.goal_index = 0
        self.goal_active = False

        self.hsv_ranges = {
            'red': [
                (np.array([0, 120, 70]), np.array([10, 255, 255])),
                (np.array([170, 120, 70]), np.array([180, 255, 255]))
            ],
            'green': [(np.array([45, 100, 100]), np.array([75, 255, 255]))],
            'blue': [(np.array([100, 100, 100]), np.array([130, 255, 255]))]
        }

    def send_next_goal(self):
        if self.goal_index >= len(self.goals) or self.close_to_blue:
            self.get_logger().info("Exploration complete or blue box found.")
            return

        x, y, yaw = self.goals[self.goal_index]
        self.goal_index += 1

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = cos(yaw / 2.0)

        self.get_logger().info(f"Navigating to: x={x}, y={y}")
        self.goal_active = True
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.get_logger().info("Goal rejected.")
            self.goal_active = False
            return

        self.get_logger().info("Goal accepted.")
        self.goal_handle.get_result_async().add_done_callback(self.goal_done_callback)

    def goal_done_callback(self, future):
        self.get_logger().info("Goal completed.")
        self.goal_active = False

    def cancel_current_goal(self):
        if self.goal_handle:
            self.goal_handle.cancel_goal_async()
            self.get_logger().info("Goal cancelled (blue detected close).")
            self.goal_active = False

    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        for colour, ranges in self.hsv_ranges.items():
            mask = None
            for lower, upper in ranges:
                this_mask = cv2.inRange(hsv, lower, upper)
                mask = this_mask if mask is None else cv2.bitwise_or(mask, this_mask)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                if area > 1000:
                    print(f"{colour.upper()} detected | Area: {area:.1f}")
                    if colour == 'blue':
                        self.seen_blue = True
                        if area > 40000:  # reasonable threshold to mean "close"
                            self.close_to_blue = True
                            self.cancel_current_goal()

        cv2.imshow("Camera View", image)
        cv2.waitKey(3)

def main():
    rclpy.init()
    node = CourseworkBot()
    node.get_logger().info("Waiting for Nav2...")
    time.sleep(5)
    node.nav_client.wait_for_server()
    node.get_logger().info("Nav2 ready.")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            if node.close_to_blue:
                node.get_logger().info("Blue object detected close — stopping.")
                break

            if not node.goal_active and not node.close_to_blue:
                node.send_next_goal()

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
