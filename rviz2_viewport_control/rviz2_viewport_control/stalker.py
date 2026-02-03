#!/usr/bin/env python3

import math
import time
from typing import Dict, Optional, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
    Marker,
    MarkerArray,
)

from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from tf2_ros import TransformBroadcaster, TransformListener, Buffer


def make_pose(x: float, y: float, z: float = 0.0) -> Pose:
    pose = Pose()
    pose.position = Point(x=x, y=y, z=z)
    pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    return pose


def quat_from_yaw(yaw_rad: float) -> Quaternion:
    half = yaw_rad * 0.5
    return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))


def bezier_cubic(p0, p1, p2, p3, t: float) -> Tuple[float, float, float]:
    """Cubic Bezier curve point in 3D."""
    u = 1.0 - t
    b0 = u * u * u
    b1 = 3.0 * u * u * t
    b2 = 3.0 * u * t * t
    b3 = t * t * t
    x = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
    y = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
    z = b0 * p0[2] + b1 * p1[2] + b2 * p2[2] + b3 * p3[2]
    return (x, y, z)


def smoothstep(t: float) -> float:
    """Smooth easing 0->1 with zero slope at ends."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


class ViewpointStalker(Node):
    """
    Subscribes to non-interactive MarkerArray from boxes.py and creates
    interactive markers for them. Handles camera transformations and TF
    connections to track the boxes.
    
    Click behavior:
        1st click: attach viewpoint to box (after smooth transition)
        2nd click: start rotation (1 rev / 120s)
        3rd click: stop + reset viewpoint to origin in map
    """

    def __init__(self):
        super().__init__("viewpoint_stalker")

        # Frames
        self.fixed_frame = "map"
        self.viewpoint_frame = "viewpoint"
        self.view_parent_frame = self.fixed_frame  # changes to selected box TF when attached

        # TF broadcaster for viewpoint
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # TF listener to get box positions
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Interactive marker server
        self.server = InteractiveMarkerServer(self, "/vis_all/interactive")

        # Subscribe to non-interactive markers from boxes
        self.marker_sub = self.create_subscription(
            MarkerArray,
            "/vis_all/all",
            self._on_marker_array,
            10
        )

        # --- Config ---
        self.box_z = 0.15
        self.rotation_period_s = 120.0
        self.transition_duration_s = 1.0

        # Track box states from TF
        self.box_positions: Dict[str, Tuple[float, float]] = {}
        self.box_yaws: Dict[str, float] = {}
        self.box_tf_frames: Dict[str, str] = {}  # marker_name -> tf frame
        self.interactive_markers_created: set = set()  # Track which markers we've created
        self.latest_marker_poses: Dict[str, Pose] = {}  # Store latest poses from MarkerArray

        # Viewpoint state (relative to view_parent_frame)
        self.view_rel_pos = (0.0, 0.0, 0.0)
        self.view_yaw = 0.0
        self.rotating = False
        self._last_rot_time = time.time()

        # Transition state
        self.transitioning = False
        self.trans_start_time = 0.0
        self.trans_from_world = (0.0, 0.0, 0.0)
        self.trans_target_box: Optional[str] = None
        self.trans_source_box: Optional[str] = None
        self.trans_from_yaw = 0.0
        self.trans_to_yaw = 0.0
        self.pending_attach_to: Optional[str] = None  # Defer frame switch to next cycle

        # Click tracking
        self.previous_clicked: Optional[str] = None
        self.selected_marker: Optional[str] = None
        self.click_stage = 0  # 0=origin, 1=attached, 2=attached+rotating

        # Track first marker reception
        self._first_markers_received = False

        # Timer for TF updates
        self.tf_timer = self.create_timer(0.02, self._publish_viewpoint_tf)  # 50 Hz

        # Start viewpoint at origin
        self._reset_viewpoint_to_origin()

        self.get_logger().info(
            "ViewpointStalker ready.\n"
            "Subscribing to: /vis_all/all\n"
            "Publishing interactive markers on namespace: /vis_all/interactive\n"
            "Publishing TF: <parent> -> viewpoint\n"
            f"Camera transition: {self.transition_duration_s:.2f}s spline"
        )

    # ---------------------------
    # Marker array subscription
    # ---------------------------
    def _on_marker_array(self, msg: MarkerArray):
        """Receives non-interactive markers and creates interactive versions."""
        if not msg.markers:
            return
        
        if not self._first_markers_received:
            self.get_logger().info(f"Received first MarkerArray with {len(msg.markers)} markers")
            self._first_markers_received = True
        
        new_markers_created = False
        
        for marker in msg.markers:
            # Extract marker name from namespace or id
            if marker.ns:
                name = marker.ns
            else:
                name = f"marker_{marker.id:02d}"
            
            # Use the marker's actual frame_id, or construct _tf frame if not provided
            if marker.header.frame_id:
                tf_frame = marker.header.frame_id
            else:
                tf_frame = f"{name}_tf"
            
            # Track the TF frame for this marker
            self.box_tf_frames[name] = tf_frame
            
            # Store latest pose for this marker
            self.latest_marker_poses[name] = marker.pose
            
            # Store position and yaw for viewpoint tracking
            x = float(marker.pose.position.x)
            y = float(marker.pose.position.y)
            self.box_positions[name] = (x, y)
            
            # Extract yaw from marker orientation
            q = marker.pose.orientation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )
            self.box_yaws[name] = float(yaw)
            
            # Check if interactive marker already exists
            if name not in self.interactive_markers_created:
                # Create new interactive marker
                im = self._make_interactive_marker_from_marker(marker, name, tf_frame)
                self.server.insert(im, feedback_callback=self._on_feedback)
                self.interactive_markers_created.add(name)
                self.get_logger().info(f"Created interactive marker: {name} on frame: {tf_frame}")
                new_markers_created = True
        
        # Only apply changes if we created new markers
        if new_markers_created:
            self.server.applyChanges()



    def _make_interactive_marker_from_marker(self, marker: Marker, name: str, tf_frame: str) -> InteractiveMarker:
        """Convert a regular Marker to an InteractiveMarker."""
        im = InteractiveMarker()
        # Use the marker's actual TF frame so it follows automatically
        im.header.frame_id = tf_frame
        im.name = name
        im.description = ""  # hide name label
        # Use the marker's actual pose relative to its frame
        im.pose = marker.pose
        im.scale = 1.0

        # Create a new marker for the control, copying properties from original
        visual_marker = Marker()
        visual_marker.type = marker.type  # Use the actual marker type (CUBE, SPHERE, etc.)
        visual_marker.scale = marker.scale
        visual_marker.color = marker.color
        visual_marker.pose = make_pose(0.0, 0.0, 0.0)  # Relative to interactive marker pose

        # Create clickable control with the marker
        control = InteractiveMarkerControl()
        control.name = "button"
        control.always_visible = True
        control.interaction_mode = InteractiveMarkerControl.BUTTON
        control.markers.append(visual_marker)

        im.controls.append(control)
        return im

    # ---------------------------
    # Helpers: get current viewpoint world pose
    # ---------------------------
    def _current_viewpoint_world(self) -> Tuple[Tuple[float, float, float], float]:
        """
        Compute viewpoint's world position from current state using TF lookups for accuracy.
        """
        if self.view_parent_frame == self.fixed_frame:
            return (self.view_rel_pos, self.view_yaw)

        # If attached to a non-map frame, use TF to get accurate world position
        try:
            # Look up the transform from map to the parent frame
            trans = self.tf_buffer.lookup_transform(
                self.fixed_frame,
                self.view_parent_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            # Get parent frame position in world
            parent_x = float(trans.transform.translation.x)
            parent_y = float(trans.transform.translation.y)
            parent_z = float(trans.transform.translation.z)
            
            # Get parent frame orientation
            q = trans.transform.rotation
            parent_yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )
            
            # Transform viewpoint relative position to world coordinates
            # Apply rotation to relative position
            cos_yaw = math.cos(parent_yaw)
            sin_yaw = math.sin(parent_yaw)
            world_x = parent_x + (self.view_rel_pos[0] * cos_yaw - self.view_rel_pos[1] * sin_yaw)
            world_y = parent_y + (self.view_rel_pos[0] * sin_yaw + self.view_rel_pos[1] * cos_yaw)
            world_z = parent_z + self.view_rel_pos[2]
            
            world_yaw = parent_yaw + self.view_yaw
            
            return ((world_x, world_y, world_z), world_yaw)
            
        except Exception as e:
            self.get_logger().warn(f"Could not lookup TF for {self.view_parent_frame}: {e}")
            # Fallback to stored position
            return (self.view_rel_pos, self.view_yaw)

    # ---------------------------
    # Viewpoint click logic + transitions
    # ---------------------------
    def _reset_viewpoint_to_origin(self):
        self.transitioning = False
        self.rotating = False
        self.view_parent_frame = self.fixed_frame
        self.view_rel_pos = (0.0, 0.0, 0.0)
        self.view_yaw = 0.0
        self._last_rot_time = time.time()
        self.click_stage = 0
        self.selected_marker = None
        self.trans_target_box = None

    def _begin_transition_to_origin(self):
        """Start a smooth transition back to world origin."""
        # Stop rotation during transition
        self.rotating = False

        # Capture current world pose
        (pos_w, yaw_w) = self._current_viewpoint_world()
        self.trans_from_world = pos_w
        self.trans_from_yaw = yaw_w

        self.get_logger().info(f"[TRANSITION] Begin transition to ORIGIN from world_pos={pos_w}, yaw={yaw_w:.2f}")

        # Target is world origin
        self.trans_target_box = None  # Special case: None means origin
        self.trans_to_yaw = 0.0

        self.transitioning = True
        self.trans_start_time = time.time()

        # During transition we publish viewpoint in map frame
        self.view_parent_frame = self.fixed_frame

        # Reset click stage
        self.click_stage = 0
        self.selected_marker = None

    def _begin_transition_to_box(self, marker_name: str):
        # Stop rotation during transition
        self.rotating = False

        # Capture current world pose
        (pos_w, yaw_w) = self._current_viewpoint_world()
        self.trans_from_world = pos_w
        self.trans_from_yaw = yaw_w
        
        self.get_logger().info(f"[TRANSITION] Begin transition to '{marker_name}' from world_pos={pos_w}, yaw={yaw_w:.2f}")
        self.get_logger().info(f"[TRANSITION] Previous parent_frame='{self.view_parent_frame}', rel_pos={self.view_rel_pos}")
        
        self.trans_target_box = marker_name
        self.trans_to_yaw = 0.0

        self.transitioning = True
        self.trans_start_time = time.time()

        # During transition we publish viewpoint in map frame
        # Set view_rel_pos to the world position we just captured
        self.view_parent_frame = self.fixed_frame
        self.view_rel_pos = pos_w
        self.view_yaw = yaw_w
        
        self.get_logger().info(f"[TRANSITION] Switched to map frame, set rel_pos={pos_w}")

        # Click stage becomes "attached pending" (we'll attach at end)
        self.click_stage = 1
        self.selected_marker = marker_name

    def _finish_attach_to_box(self):
        if self.trans_target_box is None:
            # Finishing transition to origin - just stop transitioning, stay at map
            self.transitioning = False
            self.selected_marker = None
            self.click_stage = 0
            self._last_rot_time = time.time()
            self.pending_attach_to = None
            self.get_logger().info(f"[TRANSITION] Transition to origin completed at rel_pos={self.view_rel_pos}")
            return
        
        # Finishing transition to a marker - defer the frame switch!
        target_marker = self.trans_target_box
        if target_marker not in self.box_tf_frames:
            self.get_logger().error(f"Cannot finish transition to {target_marker}: TF frame not found")
            return
        
        self.get_logger().info(f"[TRANSITION] Finishing transition to '{target_marker}'")
        self.get_logger().info(f"[TRANSITION] Current state: parent='{self.view_parent_frame}', rel_pos={self.view_rel_pos}, yaw={self.view_yaw:.2f}")
        
        self.trans_target_box = None  # Clear this FIRST to prevent re-entry
        self.transitioning = False
        self.selected_marker = target_marker
        self.click_stage = 1
        self._last_rot_time = time.time()
        
        # Defer the parent frame switch to next cycle
        self.pending_attach_to = target_marker
        self.get_logger().info(f"[TRANSITION] Set pending_attach_to='{target_marker}', will attach on next TF cycle")

    def _start_rotation(self):
        # Only makes sense when attached (not transitioning)
        self.rotating = True
        self._last_rot_time = time.time()
        self.click_stage = 2

    def _on_feedback(self, feedback: InteractiveMarkerFeedback):
        if feedback.event_type != InteractiveMarkerFeedback.BUTTON_CLICK:
            return
        
        # Handle empty feedback as "return to world origin"
        if not feedback.marker_name or feedback.marker_name == "":
            self.get_logger().info("Received return to world command")
            # Start a transition to world origin instead of snapping
            self._begin_transition_to_origin()
            return

        current = feedback.marker_name
        prev = self.previous_clicked
        self.previous_clicked = current

        # If clicking a different marker: start spline transition
        if current != self.selected_marker:
            self.get_logger().info(f"Starting transition: {self.selected_marker} -> {current}")
            self._begin_transition_to_box(current)
            return

        # Same marker clicked again:
        # If still transitioning, ignore extra clicks to avoid fighting the animation
        if self.transitioning:
            self.get_logger().info(f"Clicked: {current} (ignored during transition)")
            return

        if self.click_stage == 1:
            self._start_rotation()
            self.get_logger().info(f"Clicked: {current} | previous: {prev} (rotate ON)")
        elif self.click_stage == 2:
            self._reset_viewpoint_to_origin()
            self.get_logger().info(f"Clicked: {current} | previous: {prev} (reset)")
        else:
            # stage 0 -> treat as transition/attach
            self._begin_transition_to_box(current)
            self.get_logger().info(f"Clicked: {current} | previous: {prev} (transition)")

    # ---------------------------
    # TF publishing (smooth viewpoint)
    # ---------------------------
    def _publish_viewpoint_tf(self):
        # Handle pending attach - switch parent frame at the START of the cycle
        if self.pending_attach_to:
            target_marker = self.pending_attach_to
            self.pending_attach_to = None
            
            self.get_logger().info(f"[ATTACH] Executing pending attach to '{target_marker}'")
            
            # Capture current world position BEFORE switching frames
            current_world_pos = self.view_rel_pos  # Already in world coords since parent is map
            current_world_yaw = self.view_yaw
            
            self.get_logger().info(f"[ATTACH] Before attach: parent='{self.view_parent_frame}', world_pos={current_world_pos}, world_yaw={current_world_yaw:.2f}")
            
            # Get target TF frame's world position via TF lookup
            try:
                target_tf = self.box_tf_frames[target_marker]
                trans = self.tf_buffer.lookup_transform(
                    self.fixed_frame,
                    target_tf,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                
                # Get TF frame position and orientation
                tf_x = float(trans.transform.translation.x)
                tf_y = float(trans.transform.translation.y)
                tf_z = float(trans.transform.translation.z)
                
                q = trans.transform.rotation
                tf_yaw = math.atan2(
                    2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                )
                
                self.get_logger().info(f"[ATTACH] Target TF '{target_tf}' at world_pos=({tf_x:.2f}, {tf_y:.2f}, {tf_z:.2f}), yaw={tf_yaw:.2f}")
                
                # Calculate inverse transform: what rel_pos gives us current world pos?
                # world_pos = tf_pos + rotate(rel_pos, tf_yaw)
                # So: rel_pos = rotate_inverse(world_pos - tf_pos, tf_yaw)
                dx = current_world_pos[0] - tf_x
                dy = current_world_pos[1] - tf_y
                dz = current_world_pos[2] - tf_z
                
                self.get_logger().info(f"[ATTACH] Delta from TF: ({dx:.2f}, {dy:.2f}, {dz:.2f})")
                
                cos_yaw = math.cos(-tf_yaw)  # Inverse rotation
                sin_yaw = math.sin(-tf_yaw)
                
                rel_x = dx * cos_yaw - dy * sin_yaw
                rel_y = dx * sin_yaw + dy * cos_yaw
                rel_z = dz
                
                self.view_rel_pos = (rel_x, rel_y, rel_z)
                self.view_yaw = current_world_yaw - tf_yaw
                
                self.get_logger().info(f"[ATTACH] Calculated rel_pos=({rel_x:.2f}, {rel_y:.2f}, {rel_z:.2f}), rel_yaw={self.view_yaw:.2f}")
                
            except Exception as e:
                self.get_logger().error(f"[ATTACH] Failed to lookup TF for attach: {e}")
                self.get_logger().error(f"[ATTACH] WARNING: Falling back to marker pose - this will likely cause incorrect positioning!")
                # Fallback to marker pose
                if target_marker in self.latest_marker_poses:
                    marker_pose = self.latest_marker_poses[target_marker]
                    self.view_rel_pos = (
                        float(marker_pose.position.x),
                        float(marker_pose.position.y),
                        float(marker_pose.position.z)
                    )
                    q = marker_pose.orientation
                    marker_yaw = math.atan2(
                        2.0 * (q.w * q.z + q.x * q.y),
                        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                    )
                    self.view_yaw = marker_yaw
                else:
                    self.view_rel_pos = (0.0, 0.0, 0.0)
                    self.view_yaw = 0.0
            
            # NOW switch the parent frame
            self.view_parent_frame = self.box_tf_frames[target_marker]
            self.get_logger().info(f"[ATTACH] Frame switched: parent='{self.view_parent_frame}', rel_pos={self.view_rel_pos}, rel_yaw={self.view_yaw:.2f}")
        
        # Update viewpoint yaw if rotating
        if self.rotating:
            now = time.time()
            dt = now - self._last_rot_time
            self._last_rot_time = now
            omega = (2.0 * math.pi) / self.rotation_period_s
            self.view_yaw = (self.view_yaw + omega * dt) % (2.0 * math.pi)

        # Handle camera transition along spline (in map frame)
        if self.transitioning:
            now = time.time()
            t_raw = (now - self.trans_start_time) / max(self.transition_duration_s, 1e-6)
            t = smoothstep(t_raw)
            
            # Check if transitioning to origin or to a marker
            if self.trans_target_box is None:
                # Transitioning to world origin
                target_x, target_y, target_z = 0.0, 0.0, 0.0
                target_yaw = 0.0
            else:
                # Transitioning to a marker - get target position from TF for accuracy
                try:
                    target_tf = self.box_tf_frames.get(self.trans_target_box)
                    if not target_tf:
                        return
                    
                    trans = self.tf_buffer.lookup_transform(
                        self.fixed_frame,
                        target_tf,
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.1)
                    )
                    
                    # Get target TF position in world coordinates
                    tf_x = float(trans.transform.translation.x)
                    tf_y = float(trans.transform.translation.y)
                    tf_z = float(trans.transform.translation.z)
                    
                    # Get TF orientation
                    q = trans.transform.rotation
                    tf_yaw = math.atan2(
                        2.0 * (q.w * q.z + q.x * q.y),
                        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                    )
                    
                    # Add marker's pose offset relative to its TF frame
                    if self.trans_target_box in self.latest_marker_poses:
                        marker_pose = self.latest_marker_poses[self.trans_target_box]
                        # Transform marker position by TF rotation
                        cos_yaw = math.cos(tf_yaw)
                        sin_yaw = math.sin(tf_yaw)
                        marker_x = marker_pose.position.x
                        marker_y = marker_pose.position.y
                        marker_z = marker_pose.position.z
                        
                        target_x = tf_x + (marker_x * cos_yaw - marker_y * sin_yaw)
                        target_y = tf_y + (marker_x * sin_yaw + marker_y * cos_yaw)
                        target_z = tf_z + marker_z
                        
                        # Get marker orientation relative to TF
                        q_marker = marker_pose.orientation
                        marker_yaw = math.atan2(
                            2.0 * (q_marker.w * q_marker.z + q_marker.x * q_marker.y),
                            1.0 - 2.0 * (q_marker.y * q_marker.y + q_marker.z * q_marker.z)
                        )
                        target_yaw = tf_yaw + marker_yaw
                    else:
                        target_x = tf_x
                        target_y = tf_y
                        target_z = tf_z
                        target_yaw = tf_yaw
                    
                except Exception as e:
                    # Fallback to stored position if TF lookup fails
                    if self.trans_target_box not in self.box_positions:
                        return
                    bx, by = self.box_positions[self.trans_target_box]
                    target_x, target_y, target_z = bx, by, self.box_z
                    target_yaw = self.box_yaws.get(self.trans_target_box, 0.0)

            # Transition to target position
            p0 = self.trans_from_world
            p3 = (target_x, target_y, target_z)

            # Build a nice curved Bezier using direction vector
            dx = p3[0] - p0[0]
            dy = p3[1] - p0[1]
            dz = p3[2] - p0[2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6

            # Tangent magnitude scales with distance
            alpha = 0.35 * dist
            vx = dx / dist
            vy = dy / dist
            vz = dz / dist

            p1 = (p0[0] + vx * alpha, p0[1] + vy * alpha, p0[2] + vz * alpha)
            p2 = (p3[0] - vx * alpha, p3[1] - vy * alpha, p3[2] - vz * alpha)

            x, y, z = bezier_cubic(p0, p1, p2, p3, t)

            # Interpolate yaw to match target's current orientation
            # Handle angle wrapping for smooth interpolation
            diff = target_yaw - self.trans_from_yaw
            while diff > math.pi:
                diff -= 2.0 * math.pi
            while diff < -math.pi:
                diff += 2.0 * math.pi
            interpolated_yaw = self.trans_from_yaw + diff * t
            
            self.view_rel_pos = (float(x), float(y), float(z))
            self.view_yaw = float(interpolated_yaw)

            # End transition
            if t_raw >= 1.0:
                if self.transitioning:  # Only finish if we're still transitioning
                    self._finish_attach_to_box()

        # TF: parent -> viewpoint
        stamp = self.get_clock().now().to_msg()
        v = TransformStamped()
        v.header.stamp = stamp
        v.header.frame_id = self.view_parent_frame
        v.child_frame_id = self.viewpoint_frame
        v.transform.translation.x = float(self.view_rel_pos[0])
        v.transform.translation.y = float(self.view_rel_pos[1])
        v.transform.translation.z = float(self.view_rel_pos[2])
        v.transform.rotation = quat_from_yaw(self.view_yaw)
        self.tf_broadcaster.sendTransform(v)


def main():
    rclpy.init()
    node = ViewpointStalker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
