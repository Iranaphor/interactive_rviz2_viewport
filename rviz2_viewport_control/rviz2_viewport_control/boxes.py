#1. remove typing
#2. move utility methods to a utils.py
#3. remove _ on start of function names
#4. remove emoji
#5. subscribe to /topological_map_2 and on initial message position viewpoint tf so that the entire map is within view


#!/usr/bin/env python3

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray

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


@dataclass
class PathState:
    start: Tuple[float, float]
    ctrl: Tuple[float, float]
    goal: Tuple[float, float]
    t: float
    duration_s: float


class MovingBoxes(Node):
    """
    Publishes moving boxes as non-interactive markers and their TF frames.
    - Motion: internal sim @ 50Hz
    - Publishes MarkerArray @ 10Hz
    - TF: map -> <box>_tf for each box
    """

    def __init__(self):
        super().__init__("moving_boxes")

        # Frames
        self.fixed_frame = "map"

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Publisher for non-interactive markers
        self.marker_pub = self.create_publisher(MarkerArray, "marker_array", 10)

        # --- Config ---
        self.bounds_min = -12.0  # 24m square => [-12, +12]
        self.bounds_max = 12.0
        self.box_z = 0.15
        self.cube_size = 0.3

        # Update rates
        self.sim_dt = 0.02          # 50 Hz sim + TF
        self.marker_pub_period = 0.1 # 10 Hz marker publishing

        random.seed(7)

        # (internal_name, initial_x, initial_y, (r,g,b))
        self.box_defs = [
            ("box_00",   0.0,  0.0, (1.0, 0.1, 0.1)),    # red
            ("box_01",   2.0,  1.0, (0.1, 0.9, 0.1)),    # green
            ("box_02",   0.0,  3.0, (0.2, 0.4, 1.0)),    # blue
            ("box_03",  -2.0,  0.0, (1.0, 0.8, 0.1)),    # gold
            ("box_04",  -1.0, -2.0, (1.0, 0.2, 1.0)),    # magenta
            ("box_05",   3.0,  3.0, (0.0, 1.0, 1.0)),    # cyan
            ("box_06",  -3.0,  3.0, (1.0, 0.5, 0.0)),    # orange
            ("box_07",   4.0, -1.0, (0.5, 0.0, 1.0)),    # purple
            ("box_08",  -4.0, -3.0, (1.0, 1.0, 0.0)),    # yellow
            ("box_09",   1.0, -4.0, (0.0, 1.0, 0.5)),    # spring green
            ("box_10",  -2.0,  4.0, (1.0, 0.0, 0.5)),    # hot pink
            ("box_11",   5.0,  2.0, (0.3, 1.0, 0.3)),    # lime
            ("box_12",  -5.0,  1.0, (0.0, 0.5, 1.0)),    # sky blue
            ("box_13",   2.0, -5.0, (1.0, 0.3, 0.7)),    # pink
            ("box_14",  -1.0,  5.0, (0.7, 0.3, 1.0)),    # lavender
            ("box_15",   6.0, -2.0, (1.0, 0.6, 0.0)),    # amber
            ("box_16",  -6.0, -1.0, (0.0, 0.8, 0.8)),    # teal
            ("box_17",   3.0, -6.0, (1.0, 0.0, 0.8)),    # rose
            ("box_18",  -3.0,  6.0, (0.5, 1.0, 0.0)),    # chartreuse
            ("box_19",   7.0,  3.0, (0.0, 0.6, 1.0)),    # azure
        ]

        # State: box motion
        self.box_tf: Dict[str, str] = {}                  # marker_name -> tf frame
        self.box_pos: Dict[str, Tuple[float, float]] = {} # marker_name -> (x,y)
        self.box_paths: Dict[str, PathState] = {}         # marker_name -> path
        self.box_yaw: Dict[str, float] = {}               # marker_name -> current orientation
        self.box_colors: Dict[str, Tuple[float, float, float]] = {}  # marker_name -> (r,g,b)

        # Initialize boxes
        for name, x, y, rgb in self.box_defs:
            self.box_tf[name] = f"{name}_tf"
            self.box_pos[name] = (x, y)
            self.box_paths[name] = self._new_path_from((x, y))
            self.box_yaw[name] = 0.0
            self.box_colors[name] = rgb

        # Timers
        self.sim_timer = self.create_timer(self.sim_dt, self._step_motion)
        self.marker_timer = self.create_timer(self.marker_pub_period, self._publish_markers)
        self.tf_timer = self.create_timer(self.sim_dt, self._publish_tfs)

        # Track if we've logged marker publishing
        self._markers_published_once = False

        self.get_logger().info(
            "MovingBoxes ready.\n"
            "Publishing MarkerArray to: marker_array\n"
            "Publishing TFs: map -> <box>_tf for each box\n"
            "Fixed frame: map"
        )

    # ---------------------------
    # Marker publishing
    # ---------------------------
    def _publish_markers(self):
        """Publish non-interactive markers for all boxes."""
        marker_array = MarkerArray()
        
        for name, (x, y) in self.box_pos.items():
            marker = Marker()
            marker.header.frame_id = self.fixed_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = name
            marker.id = 0
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Position and orientation
            marker.pose = make_pose(x, y, self.box_z)
            marker.pose.orientation = quat_from_yaw(self.box_yaw[name])
            
            # Scale
            marker.scale.x = self.cube_size
            marker.scale.y = self.cube_size
            marker.scale.z = self.cube_size
            
            # Color
            rgb = self.box_colors[name]
            marker.color.r = float(rgb[0])
            marker.color.g = float(rgb[1])
            marker.color.b = float(rgb[2])
            marker.color.a = 1.0
            
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)
        
        if not self._markers_published_once:
            self.get_logger().info(f"Published {len(marker_array.markers)} markers to marker_array topic")
            self._markers_published_once = True

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
            old_pos = self.box_pos[name]
            path.t += dt / max(path.duration_s, 1e-6)

            if path.t >= 1.0:
                self.box_pos[name] = path.goal
                new_path = self._new_path_from(self.box_pos[name])
                self.box_paths[name] = new_path
                # Calculate direction of new path for smooth rotation
                dx = new_path.goal[0] - new_path.start[0]
                dy = new_path.goal[1] - new_path.start[1]
            else:
                x, y = bezier_quad(path.start, path.ctrl, path.goal, path.t)
                self.box_pos[name] = (x, y)
                # Compute target yaw from velocity direction
                dx = x - old_pos[0]
                dy = y - old_pos[1]
            
            # Update yaw smoothly for both cases
            if abs(dx) > 1e-4 or abs(dy) > 1e-4:
                target_yaw = math.atan2(dy, dx)
                # Smoothly interpolate current yaw toward target
                current_yaw = self.box_yaw[name]
                # Handle angle wrapping
                diff = target_yaw - current_yaw
                while diff > math.pi:
                    diff -= 2.0 * math.pi
                while diff < -math.pi:
                    diff += 2.0 * math.pi
                # Interpolate with damping factor (adjust 3.0 for faster/slower rotation)
                self.box_yaw[name] = current_yaw + diff * min(1.0, dt * 3.0)

    # ---------------------------
    # TF publishing
    # ---------------------------
    def _publish_tfs(self):
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
            t.transform.rotation = quat_from_yaw(self.box_yaw[name])
            self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = MovingBoxes()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
