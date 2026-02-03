#!/usr/bin/env python3

import random
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import InteractiveMarkerFeedback, MarkerArray


class AutoFollower(Node):
    """
    Automatically selects random markers to follow every 10 seconds.
    Subscribes to marker_array to discover available markers dynamically.
    Listens to /vis_all/target to follow specific targets with auto mode pause.
    Publishes feedback messages to trigger the stalker node to transition
    between markers.
    """

    def __init__(self):
        super().__init__("auto_follower")

        # Publisher for interactive marker feedback
        self.feedback_pub = self.create_publisher(
            InteractiveMarkerFeedback,
            "/vis_all/interactive/feedback",
            10
        )

        # Subscribe to marker_array to discover markers
        self.marker_sub = self.create_subscription(
            MarkerArray,
            "/vis_all/all",
            self._on_marker_array,
            10
        )
        
        # Subscribe to target requests
        self.target_sub = self.create_subscription(
            String,
            "/vis_all/target",
            self._on_target_request,
            10
        )

        # Track available markers
        self.available_markers = set()
        self.last_selected_marker = None
        
        # Timer control
        self.timer = None
        self.timer_enabled = True

        random.seed()  # Use system time for randomness

        # Create timer for auto-clicking (10 seconds)
        self._create_timer()

        self.get_logger().info(
            "AutoFollower ready.\n"
            "Subscribing to: /vis_all/all\n"
            "Subscribing to: /vis_all/target\n"
            "Publishing feedback to: /vis_all/interactive/feedback\n"
            "Interval: 10 seconds"
        )

    def _on_marker_array(self, msg: MarkerArray):
        """Update the list of available markers from the marker array."""
        for marker in msg.markers:
            # Extract marker name from namespace or id
            if marker.ns:
                name = marker.ns
            else:
                name = f"marker_{marker.id:02d}"
            
            self.available_markers.add(name)
    
    def _create_timer(self):
        """Create the auto-click timer."""
        if self.timer:
            self.timer.cancel()
        self.timer = self.create_timer(10.0, self._auto_click_marker)
        self.timer_enabled = True
    
    def _stop_timer(self):
        """Stop the auto-click timer."""
        if self.timer:
            self.timer.cancel()
            self.timer_enabled = False
    
    def _on_target_request(self, msg: String):
        """Handle target request from /vis_all/target topic."""
        frame_id = msg.data.strip()
        if not frame_id:
            self.get_logger().warn("Received empty target frame_id")
            return
        
        self.get_logger().info(f"Target request received: {frame_id}")
        
        # Stop auto timer
        self._stop_timer()
        
        # Find marker with frame_id + "/base_link"
        target_frame = f"{frame_id}/base_link"
        matching_marker = None
        
        for marker_name in self.available_markers:
            # Check if this marker matches the target frame
            # Marker names might be the full namespace which could match
            if target_frame in marker_name or marker_name.startswith(frame_id):
                matching_marker = marker_name
                break
        
        if matching_marker:
            self.get_logger().info(f"Found matching marker: {matching_marker}")
            
            # Send feedback to follow this marker
            feedback = InteractiveMarkerFeedback()
            feedback.header.stamp = self.get_clock().now().to_msg()
            feedback.header.frame_id = "map"
            feedback.marker_name = matching_marker
            feedback.event_type = InteractiveMarkerFeedback.BUTTON_CLICK
            self.feedback_pub.publish(feedback)
            
            # Sleep for 10 seconds
            self.get_logger().info("Waiting 10 seconds before returning to world...")
            time.sleep(10.0)
            
            # Send empty feedback to return to world
            self.get_logger().info("Sending return to world command")
            empty_feedback = InteractiveMarkerFeedback()
            empty_feedback.header.stamp = self.get_clock().now().to_msg()
            empty_feedback.header.frame_id = ""
            empty_feedback.marker_name = ""
            empty_feedback.event_type = InteractiveMarkerFeedback.BUTTON_CLICK
            self.feedback_pub.publish(empty_feedback)
            
            # Re-enable timer
            self.get_logger().info("Re-enabling auto-follow timer")
            self._create_timer()
        else:
            self.get_logger().warn(f"No marker found for target: {target_frame}")
            # Re-enable timer anyway
            self._create_timer()

    def _auto_click_marker(self):
        """Automatically click a random marker every 10 seconds."""
        if not self.timer_enabled:
            return
            
        if not self.available_markers:
            self.get_logger().warn("No markers available yet")
            return

        # Create a pool of markers excluding the last selected one
        available_pool = list(self.available_markers)
        if self.last_selected_marker and self.last_selected_marker in available_pool and len(available_pool) > 1:
            available_pool.remove(self.last_selected_marker)

        random_marker = random.choice(available_pool)
        self.last_selected_marker = random_marker

        # Create feedback message
        feedback = InteractiveMarkerFeedback()
        feedback.header.stamp = self.get_clock().now().to_msg()
        feedback.header.frame_id = "map"
        feedback.marker_name = random_marker
        feedback.event_type = InteractiveMarkerFeedback.BUTTON_CLICK

        # Publish the feedback
        self.feedback_pub.publish(feedback)

        self.get_logger().info(f"Auto-clicked: {random_marker} (from {len(self.available_markers)} available)")


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
