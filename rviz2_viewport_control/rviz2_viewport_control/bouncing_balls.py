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


@dataclass
class BallState:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    radius: float
    color: Tuple[float, float, float]
    energy: float  # Track energy for boost detection


class BouncingBalls(Node):
    """
    Publishes bouncing balls as non-interactive markers with semi-realistic physics.
    Balls bounce with energy loss but get boosted when bounces diminish too much.
    """

    def __init__(self):
        super().__init__("bouncing_balls")

        # Frames
        self.fixed_frame = "map"

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Publisher for non-interactive markers
        self.marker_pub = self.create_publisher(MarkerArray, "/vis_all/all", 10)

        # --- Config ---
        self.bounds_min = -12.0
        self.bounds_max = 12.0
        self.ground_z = 0.0
        self.ceiling_z = 5.0

        # Physics
        self.gravity = -9.8  # m/s^2
        self.restitution = 0.75  # Coefficient of restitution (energy retained on bounce)
        self.restitution_variability = 0.1  # Random variation in bounce
        self.min_energy_threshold = 2.0  # When to boost
        self.boost_energy = 20.0  # Energy added on boost

        # Update rates
        self.sim_dt = 0.02          # 50 Hz sim + TF
        self.marker_pub_period = 0.1 # 10 Hz marker publishing

        random.seed(42)

        # (internal_name, initial_x, initial_y, initial_z, (r,g,b), radius)
        self.ball_defs = [
            ("ball_00",  -8.0,  -8.0, 2.0, (1.0, 0.3, 0.3), 0.25),  # red
            ("ball_01",  -6.0,  -6.0, 2.5, (0.3, 1.0, 0.3), 0.3),   # green
            ("ball_02",  -4.0,  -4.0, 3.0, (0.3, 0.5, 1.0), 0.2),   # blue
            ("ball_03",  -2.0,  -2.0, 2.2, (1.0, 1.0, 0.3), 0.28),  # yellow
            ("ball_04",   0.0,   0.0, 2.8, (1.0, 0.3, 1.0), 0.22),  # magenta
            ("ball_05",   2.0,   2.0, 2.4, (0.3, 1.0, 1.0), 0.26),  # cyan
            ("ball_06",   4.0,   4.0, 3.2, (1.0, 0.6, 0.2), 0.24),  # orange
            ("ball_07",   6.0,   6.0, 2.6, (0.6, 0.3, 1.0), 0.27),  # purple
            ("ball_08",   8.0,   8.0, 2.9, (0.3, 0.8, 0.5), 0.21),  # teal
            ("ball_09",  -7.0,   7.0, 3.1, (1.0, 0.4, 0.6), 0.29),  # pink
        ]

        # State: ball physics
        self.ball_states: Dict[str, BallState] = {}
        self.ball_tf: Dict[str, str] = {}

        # Initialize balls
        for name, x, y, z, rgb, radius in self.ball_defs:
            self.ball_tf[name] = f"{name}_tf"
            # Start with random velocity
            vx = random.uniform(-2.0, 2.0)
            vy = random.uniform(-2.0, 2.0)
            vz = random.uniform(0.0, 3.0)
            initial_energy = 0.5 * (vx*vx + vy*vy + vz*vz) + abs(self.gravity) * z
            self.ball_states[name] = BallState(
                x=x, y=y, z=z,
                vx=vx, vy=vy, vz=vz,
                radius=radius,
                color=rgb,
                energy=initial_energy
            )

        # Timers
        self.sim_timer = self.create_timer(self.sim_dt, self._step_physics)
        self.marker_timer = self.create_timer(self.marker_pub_period, self._publish_markers)
        self.tf_timer = self.create_timer(self.sim_dt, self._publish_tfs)

        # Track if we've logged marker publishing
        self._markers_published_once = False

        self.get_logger().info(
            "BouncingBalls ready.\n"
            "Publishing MarkerArray to: /vis_all/all\n"
            "Publishing TFs: map -> <ball>_tf for each ball\n"
            "Fixed frame: map"
        )

    # ---------------------------
    # Marker publishing
    # ---------------------------
    def _publish_markers(self):
        """Publish non-interactive markers for all balls."""
        marker_array = MarkerArray()
        
        for name, state in self.ball_states.items():
            marker = Marker()
            marker.header.frame_id = self.fixed_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = name
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # Position (no orientation for spheres)
            marker.pose = make_pose(state.x, state.y, state.z)
            
            # Scale
            marker.scale.x = state.radius * 2.0
            marker.scale.y = state.radius * 2.0
            marker.scale.z = state.radius * 2.0
            
            # Color
            marker.color.r = float(state.color[0])
            marker.color.g = float(state.color[1])
            marker.color.b = float(state.color[2])
            marker.color.a = 0.9
            
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)
        
        if not self._markers_published_once:
            self.get_logger().info(f"Published {len(marker_array.markers)} ball markers to /vis_all/all topic")
            self._markers_published_once = True

    # ---------------------------
    # Physics simulation
    # ---------------------------
    def _step_physics(self):
        dt = self.sim_dt
        
        for name, state in self.ball_states.items():
            # Apply gravity
            state.vz += self.gravity * dt
            
            # Update position
            state.x += state.vx * dt
            state.y += state.vy * dt
            state.z += state.vz * dt
            
            # Calculate current kinetic + potential energy
            kinetic = 0.5 * (state.vx*state.vx + state.vy*state.vy + state.vz*state.vz)
            potential = abs(self.gravity) * max(0, state.z - self.ground_z)
            state.energy = kinetic + potential
            
            # Bounce off ground
            if state.z - state.radius < self.ground_z:
                state.z = self.ground_z + state.radius
                # Apply restitution with variability
                bounce_factor = self.restitution + random.uniform(-self.restitution_variability, self.restitution_variability)
                bounce_factor = max(0.5, min(0.95, bounce_factor))  # Clamp to reasonable range
                state.vz = -state.vz * bounce_factor
                
                # Add slight random horizontal impulse on bounce
                state.vx += random.uniform(-0.5, 0.5)
                state.vy += random.uniform(-0.5, 0.5)
            
            # Bounce off ceiling
            if state.z + state.radius > self.ceiling_z:
                state.z = self.ceiling_z - state.radius
                bounce_factor = self.restitution + random.uniform(-self.restitution_variability, self.restitution_variability)
                bounce_factor = max(0.5, min(0.95, bounce_factor))
                state.vz = -state.vz * bounce_factor
            
            # Bounce off walls (x direction)
            if state.x - state.radius < self.bounds_min:
                state.x = self.bounds_min + state.radius
                bounce_factor = self.restitution + random.uniform(-self.restitution_variability, self.restitution_variability)
                bounce_factor = max(0.5, min(0.95, bounce_factor))
                state.vx = -state.vx * bounce_factor
            elif state.x + state.radius > self.bounds_max:
                state.x = self.bounds_max - state.radius
                bounce_factor = self.restitution + random.uniform(-self.restitution_variability, self.restitution_variability)
                bounce_factor = max(0.5, min(0.95, bounce_factor))
                state.vx = -state.vx * bounce_factor
            
            # Bounce off walls (y direction)
            if state.y - state.radius < self.bounds_min:
                state.y = self.bounds_min + state.radius
                bounce_factor = self.restitution + random.uniform(-self.restitution_variability, self.restitution_variability)
                bounce_factor = max(0.5, min(0.95, bounce_factor))
                state.vy = -state.vy * bounce_factor
            elif state.y + state.radius > self.bounds_max:
                state.y = self.bounds_max - state.radius
                bounce_factor = self.restitution + random.uniform(-self.restitution_variability, self.restitution_variability)
                bounce_factor = max(0.5, min(0.95, bounce_factor))
                state.vy = -state.vy * bounce_factor
            
            # Apply air resistance
            drag = 0.995
            state.vx *= drag
            state.vy *= drag
            
            # Check if energy is too low and boost
            if state.energy < self.min_energy_threshold:
                # Give a boost in a random direction
                boost_angle_xy = random.uniform(0, 2 * math.pi)
                boost_angle_z = random.uniform(math.pi/6, math.pi/3)  # 30-60 degrees up
                
                boost_speed = math.sqrt(2.0 * self.boost_energy)
                state.vx = boost_speed * math.cos(boost_angle_xy) * math.cos(boost_angle_z)
                state.vy = boost_speed * math.sin(boost_angle_xy) * math.cos(boost_angle_z)
                state.vz = boost_speed * math.sin(boost_angle_z)
                
                state.energy = self.boost_energy

    # ---------------------------
    # TF publishing
    # ---------------------------
    def _publish_tfs(self):
        stamp = self.get_clock().now().to_msg()

        # TF: map -> ball_tf
        for name, state in self.ball_states.items():
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.fixed_frame
            t.child_frame_id = self.ball_tf[name]
            t.transform.translation.x = float(state.x)
            t.transform.translation.y = float(state.y)
            t.transform.translation.z = float(state.z)
            t.transform.rotation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = BouncingBalls()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
