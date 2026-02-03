#!/usr/bin/env python3

import random
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import InteractiveMarkerFeedback


class AutoFollower(Node):
    """
    Automatically selects random boxes to follow every 30 seconds.
    Publishes feedback messages to trigger the stalker node to transition
    between boxes.
    """

    def __init__(self):
        super().__init__("auto_follower")

        # Publisher for interactive marker feedback
        self.feedback_pub = self.create_publisher(
            InteractiveMarkerFeedback,
            "/interactive_boxes/feedback",
            10
        )

        # Configuration
        self.box_names = [
            "box_00", "box_01", "box_02", "box_03", "box_04",
            "box_05", "box_06", "box_07", "box_08", "box_09",
            "box_10", "box_11", "box_12", "box_13", "box_14",
            "box_15", "box_16", "box_17", "box_18", "box_19",
        ]

        random.seed()  # Use system time for randomness

        # Timer for auto-clicking
        self.timer = self.create_timer(30.0, self._auto_click_box)

        self.get_logger().info(
            "AutoFollower ready.\n"
            "Publishing feedback to: /interactive_boxes/feedback\n"
            "Interval: 30 seconds"
        )

    def _auto_click_box(self):
        """Automatically click a random box every 30 seconds."""
        if not self.box_names:
            return

        random_box = random.choice(self.box_names)

        # Create feedback message
        feedback = InteractiveMarkerFeedback()
        feedback.header.stamp = self.get_clock().now().to_msg()
        feedback.header.frame_id = "map"
        feedback.marker_name = random_box
        feedback.event_type = InteractiveMarkerFeedback.BUTTON_CLICK

        # Publish the feedback
        self.feedback_pub.publish(feedback)

        self.get_logger().info(f"Auto-clicked: {random_box}")


def main():
    rclpy.init()
    node = AutoFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
