#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

class SunGlow(Node):

    def __init__(self):
        super().__init__("sun_glow")

        self.pub = self.create_publisher(MarkerArray, "/sun_glow", 10)
        self.t = 0.0
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.update)  # 20 Hz

    def make_sphere(self, mid, frame, x, y, z, d, r, g, b, a):
        m = Marker()
        m.header.frame_id = frame
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.ns = "sun"
        m.id = mid
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z
        m.scale.x = d
        m.scale.y = d
        m.scale.z = d
        m.color.r = r
        m.color.g = g
        m.color.b = b
        m.color.a = a
        m.lifetime.sec = 0
        return m

    def make_ring(self, mid, frame, x, y, z, radius, width, r, g, b, a, n=96):
        m = Marker()
        m.header.frame_id = frame
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.ns = "sun"
        m.id = mid
        m.pose.position.x = 0.0
        m.pose.position.y = 0.0
        m.pose.position.z = 0.0
        m.scale.x = width
        m.color.r = r
        m.color.g = g
        m.color.b = b
        m.color.a = a
        m.lifetime.sec = 0

        m.points = []
        for i in range(n + 1):
            th = 2.0 * math.pi * (i / n)
            p = type(m.points[0])() if m.points else None  # avoid importing geometry_msgs
            # quick way: use the same Point class RViz gives us
            from geometry_msgs.msg import Point
            p = Point()
            p.x = x + radius * math.cos(th)
            p.y = y + radius * math.sin(th)
            p.z = z
            m.points.append(p)
        return m

    def update(self):
        self.t += self.dt

        # pulse in [0,1]
        pulse = 0.5 + 0.5 * math.sin(self.t * 1.2)
        pulse2 = 0.5 + 0.5 * math.sin(self.t * 2.6 + 1.0)

        x, y, z = 70.0, -30.0, -200.0
        frame = "map"

        core_d = 100.0
        core_d *= (0.98 + 0.04 * pulse)  # slight breathing

        # Yellow sun-ish core
        core_r, core_g, core_b = 1.0, 0.92, 0.25

        arr = MarkerArray()

        # Core: fully opaque, bright
        arr.markers.append(self.make_sphere(
            0, frame, x, y, z,
            core_d,
            core_r, core_g, core_b,
            1.0
        ))

        # Halo layers: make them MUCH bigger, with very low alpha
        # (This is where the "glow" illusion comes from.)
        halos = [
            (1.25, 0.20),
            (1.60, 0.10),
            (2.10, 0.06),
            (2.80, 0.035),
            (3.60, 0.020),
        ]

        for i, (mul, abase) in enumerate(halos, start=1):
            d = core_d * mul * (0.90 + 0.20 * pulse)  # halos breathe more than core
            a = abase * (0.50 + 0.90 * pulse2)       # pulse alpha

            # warm corona colour shifts outward slightly orange
            r = 1.0
            g = 0.85 - 0.10 * (i / len(halos))
            b = 0.20 - 0.05 * (i / len(halos))

            # tiny jitter to stop it feeling perfectly static
            j = 0.15 * mul
            zx = x + j * math.sin(self.t * (1.0 + 0.3*i) + i)
            zy = y + j * math.cos(self.t * (0.9 + 0.25*i) + 2*i)
            zz = z + 0.002 * i  # z-fighting guard

            arr.markers.append(self.make_sphere(
                i, frame, zx, zy, zz,
                d,
                r, g, b,
                a
            ))

        # Corona ring: a subtle expanding ring gives "sun activity"
        # (LINE_STRIP is surprisingly effective for this.)
        ring_radius = (core_d * 0.65) * (1.00 + 0.12 * pulse)
        ring_width = 2.5 * (1.00 + 0.30 * pulse2)
        ring_alpha = 0.35 * (0.40 + 0.60 * pulse2)

        # NOTE: this imports geometry_msgs Point inside make_ring to avoid extra imports at top
        arr.markers.append(self.make_ring(
            100, frame, x, y, z,
            ring_radius,
            ring_width,
            1.0, 0.85, 0.15,
            ring_alpha
        ))

        self.pub.publish(arr)


def main():
    rclpy.init()
    node = SunGlow()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
