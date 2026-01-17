#1. remove typing
#2. move utility methods to a utils.py
#3. remove _ on start of function names
#4. remove emoji
#5. subscribe to /topological_map_2 and on initial message position viewpoint tf so that the entire map is within view


#!/usr/bin/env python3

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
    Marker,
)

from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from tf2_ros import TransformBroadcaster


def make_pose(x: float, y: float, z: float = 0.0) -> Pose:
    pose = Pose()
    pose.position = Point(x=x, y=y, z=z)
    pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    return pose


def quat_from_yaw(yaw_rad: float) -> Quaternion:
    half = yaw_rad * 0.5
    return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))


def bezier_quad(p0, p1, p2, t: float) -> Tuple[float, float]:
    """Quadratic Bezier curve point."""
    u = 1.0 - t
    x = u * u * p0[0] + 2.0 * u * t * p1[0] + t * t * p2[0]
    y = u * u * p0[1] + 2.0 * u * t * p1[1] + t * t * p2[1]
    return (x, y)


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


@dataclass
class PathState:
    start: Tuple[float, float]
    ctrl: Tuple[float, float]
    goal: Tuple[float, float]
    t: float
    duration_s: float


class InteractiveBoxes(Node):
    """
    - Moving INTERACTIVE markers (clickable while moving)
    - Motion: internal sim @ 50Hz
    - Interactive marker pose updates throttled @ 10Hz (reduces RViz sequence issues)
    - TF: map -> <box>_tf (moving), and <parent> -> viewpoint
    - Viewpoint transitions between boxes via spline rather than snapping
    - Click cycle on SAME box:
        1st click: attach viewpoint to box (after transition)
        2nd click: start rotation (1 rev / 12s)
        3rd click: stop + reset viewpoint to origin in map
    """

    def __init__(self):
        super().__init__("interactive_boxes")

        # Frames
        self.fixed_frame = "map"
        self.viewpoint_frame = "viewpoint"
        self.view_parent_frame = self.fixed_frame  # changes to selected box TF when attached

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Interactive marker server
        self.server = InteractiveMarkerServer(self, "interactive_boxes")

        # --- Config ---
        self.bounds_min = -5.0  # 10m square => [-5, +5]
        self.bounds_max = 5.0
        self.box_z = 0.15
        self.cube_size = 0.3

        # Rotation: 1 revolution per 12 seconds ✅
        self.rotation_period_s = 120.0

        # Update rates
        self.sim_dt = 0.02          # 50 Hz sim + TF
        self.im_update_period = 0.1 # 10 Hz interactive marker updates

        # Camera transition
        self.transition_duration_s = 1.0  # 🧠 tweak this for slower/faster camera moves

        random.seed(7)

        # (internal_name, initial_x, initial_y, (r,g,b))
        self.box_defs = [
            ("box_00",   0.0,  0.0, (1.0, 0.1, 0.1)),   # red
            ("box_10",   1.0,  0.0, (0.1, 0.9, 0.1)),   # green
            ("box_01",   0.0,  1.0, (0.2, 0.4, 1.0)),   # blue
            ("box_m10", -1.0,  0.0, (1.0, 0.8, 0.1)),   # gold
            ("box_m1m1",-1.0, -1.0, (1.0, 0.2, 1.0)),   # magenta
        ]

        # State: box motion
        self.box_tf: Dict[str, str] = {}                  # marker_name -> tf frame
        self.box_pos: Dict[str, Tuple[float, float]] = {} # marker_name -> (x,y)
        self.box_paths: Dict[str, PathState] = {}         # marker_name -> path

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
        self.trans_from_yaw = 0.0
        self.trans_to_yaw = 0.0

        # Click tracking
        self.previous_clicked: Optional[str] = None
        self.selected_marker: Optional[str] = None
        self.click_stage = 0  # 0=origin, 1=attached, 2=attached+rotating

        # Create interactive markers (clickable + visible)
        for name, x, y, rgb in self.box_defs:
            self.box_tf[name] = f"{name}_tf"
            self.box_pos[name] = (x, y)
            self.box_paths[name] = self._new_path_from((x, y))

            im = self._make_box_interactive_marker(name=name, x=x, y=y, rgb=rgb)
            self.server.insert(im, feedback_callback=self._on_feedback)

        self.server.applyChanges()

        # Timers
        self.sim_timer = self.create_timer(self.sim_dt, self._step_motion)
        self.im_timer = self.create_timer(self.im_update_period, self._update_interactive_marker_poses)
        self.tf_timer = self.create_timer(self.sim_dt, self._publish_tfs)

        # Start viewpoint at origin
        self._reset_viewpoint_to_origin()

        self.get_logger().info(
            "✅ Ready.\n"
            "RViz InteractiveMarkers Topic Namespace: interactive_boxes\n"
            "Fixed frame: map\n"
            "TFs: map -> <box>_tf and <parent> -> viewpoint\n"
            f"Camera transition: {self.transition_duration_s:.2f}s spline"
        )

    # ---------------------------
    # Interactive marker creation
    # ---------------------------
    def _make_box_interactive_marker(self, name: str, x: float, y: float, rgb) -> InteractiveMarker:
        im = InteractiveMarker()
        im.header.frame_id = self.fixed_frame
        im.name = name
        im.description = ""  # hide name label ✅
        im.pose = make_pose(x, y, self.box_z)
        im.scale = 1.0

        cube = Marker()
        cube.type = Marker.CUBE
        cube.scale.x = self.cube_size
        cube.scale.y = self.cube_size
        cube.scale.z = self.cube_size
        cube.color.r = float(rgb[0])
        cube.color.g = float(rgb[1])
        cube.color.b = float(rgb[2])
        cube.color.a = 1.0  # bold/opaque ✅

        control = InteractiveMarkerControl()
        control.name = ""
        control.always_visible = True
        control.interaction_mode = InteractiveMarkerControl.BUTTON
        control.markers.append(cube)

        im.controls.append(control)
        return im

    # ---------------------------
    # Motion generation
    # ---------------------------
    def _rand_point(self) -> Tuple[float, float]:
        return (
            random.uniform(self.bounds_min, self.bounds_max),
            random.uniform(self.bounds_min, self.bounds_max),
        )

    def _new_path_from(self, start_xy: Tuple[float, float]) -> PathState:
        goal = self._rand_point()

        mx = 0.5 * (start_xy[0] + goal[0])
        my = 0.5 * (start_xy[1] + goal[1])

        dx = goal[0] - start_xy[0]
        dy = goal[1] - start_xy[1]
        length = math.hypot(dx, dy) + 1e-6

        # perpendicular unit vector
        px = -dy / length
        py = dx / length

        curve = random.uniform(0.5, 2.5) * (1.0 if random.random() < 0.5 else -1.0)
        ctrl = (mx + px * curve, my + py * curve)

        duration_s = random.uniform(6.0, 12.0)
        return PathState(start=start_xy, ctrl=ctrl, goal=goal, t=0.0, duration_s=duration_s)

    def _step_motion(self):
        dt = self.sim_dt
        for name, path in list(self.box_paths.items()):
            path.t += dt / max(path.duration_s, 1e-6)

            if path.t >= 1.0:
                self.box_pos[name] = path.goal
                self.box_paths[name] = self._new_path_from(self.box_pos[name])
                continue

            x, y = bezier_quad(path.start, path.ctrl, path.goal, path.t)
            self.box_pos[name] = (x, y)

    # ---------------------------
    # Interactive marker pose updates (throttled)
    # ---------------------------
    def _update_interactive_marker_poses(self):
        for name, (x, y) in self.box_pos.items():
            self.server.setPose(name, make_pose(x, y, self.box_z))
        self.server.applyChanges()

    # ---------------------------
    # Helpers: get current viewpoint world pose
    # ---------------------------
    def _current_viewpoint_world(self) -> Tuple[Tuple[float, float, float], float]:
        """
        Since viewpoint is always either:
        - parent=map at some (x,y,z), or
        - parent=box_tf with rel (0,0,0)
        we can compute its world position without TF lookup.
        """
        if self.view_parent_frame == self.fixed_frame:
            return (self.view_rel_pos, self.view_yaw)

        # If attached to a box tf
        # Find which box frame we're attached to
        attached_box = None
        for name, tf_name in self.box_tf.items():
            if tf_name == self.view_parent_frame:
                attached_box = name
                break

        if attached_box is None:
            # Fallback
            return (self.view_rel_pos, self.view_yaw)

        bx, by = self.box_pos[attached_box]
        return ((bx, by, self.box_z), self.view_yaw)

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

    def _begin_transition_to_box(self, marker_name: str):
        # Stop rotation during transition
        self.rotating = False

        # Capture current world pose
        (pos_w, yaw_w) = self._current_viewpoint_world()
        self.trans_from_world = pos_w
        self.trans_from_yaw = yaw_w

        self.trans_target_box = marker_name
        self.trans_to_yaw = 0.0

        self.transitioning = True
        self.trans_start_time = time.time()

        # During transition we publish viewpoint in map frame
        self.view_parent_frame = self.fixed_frame

        # Click stage becomes "attached pending" (we'll attach at end)
        self.click_stage = 1
        self.selected_marker = marker_name

    def _finish_attach_to_box(self):
        if self.trans_target_box is None:
            return
        self.view_parent_frame = self.box_tf[self.trans_target_box]
        self.view_rel_pos = (0.0, 0.0, 0.0)
        self.view_yaw = 0.0
        self.transitioning = False
        self.trans_target_box = None
        self._last_rot_time = time.time()

    def _start_rotation(self):
        # Only makes sense when attached (not transitioning)
        self.rotating = True
        self._last_rot_time = time.time()
        self.click_stage = 2

    def _on_feedback(self, feedback: InteractiveMarkerFeedback):
        if feedback.event_type != InteractiveMarkerFeedback.BUTTON_CLICK:
            return

        current = feedback.marker_name
        prev = self.previous_clicked
        self.previous_clicked = current

        # If clicking a different marker: start spline transition
        if current != self.selected_marker:
            self._begin_transition_to_box(current)
            print(f"[interactive_boxes] clicked: {current} | previous: {prev}  (transition)")
            return

        # Same marker clicked again:
        # If still transitioning, ignore extra clicks to avoid fighting the animation
        if self.transitioning:
            print(f"[interactive_boxes] clicked: {current} (ignored during transition)")
            return

        if self.click_stage == 1:
            self._start_rotation()
            print(f"[interactive_boxes] clicked: {current} | previous: {prev}  (rotate ON)")
        elif self.click_stage == 2:
            self._reset_viewpoint_to_origin()
            print(f"[interactive_boxes] clicked: {current} | previous: {prev}  (reset)")
        else:
            # stage 0 -> treat as transition/attach
            self._begin_transition_to_box(current)
            print(f"[interactive_boxes] clicked: {current} | previous: {prev}  (transition)")

    # ---------------------------
    # TF publishing (smooth)
    # ---------------------------
    def _publish_tfs(self):
        # Update viewpoint yaw if rotating
        if self.rotating:
            now = time.time()
            dt = now - self._last_rot_time
            self._last_rot_time = now
            omega = (2.0 * math.pi) / self.rotation_period_s
            self.view_yaw = (self.view_yaw + omega * dt) % (2.0 * math.pi)

        # Handle camera transition along spline (in map frame)
        if self.transitioning and self.trans_target_box is not None:
            now = time.time()
            t_raw = (now - self.trans_start_time) / max(self.transition_duration_s, 1e-6)
            t = smoothstep(t_raw)

            # Target is the CURRENT box position (so camera homes onto moving box)
            bx, by = self.box_pos[self.trans_target_box]
            p0 = self.trans_from_world
            p3 = (bx, by, self.box_z)

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

            # Interpolate yaw back to 0 during transition
            self.view_rel_pos = (x, y, z)
            self.view_yaw = (1.0 - t) * self.trans_from_yaw + t * self.trans_to_yaw

            # End transition
            if t_raw >= 1.0:
                self._finish_attach_to_box()

        stamp = self.get_clock().now().to_msg()

        # TF: map -> box_tf (moving)
        for name, (x, y) in self.box_pos.items():
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.fixed_frame
            t.child_frame_id = self.box_tf[name]
            t.transform.translation.x = float(x)
            t.transform.translation.y = float(y)
            t.transform.translation.z = float(self.box_z)
            t.transform.rotation = quat_from_yaw(0.0)
            self.tf_broadcaster.sendTransform(t)

        # TF: parent -> viewpoint
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
    node = InteractiveBoxes()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
