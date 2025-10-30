#!/usr/bin/env python3
import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_dilation

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.exceptions import ParameterAlreadyDeclaredException
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time
from rclpy.duration import Duration

from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Pose, Quaternion

import tf2_ros
from tf2_ros import TransformException
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw / 2.0); q.z = math.sin(yaw / 2.0)
    q.x = 0.0; q.y = 0.0
    return q


class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')

        try:
            self.declare_parameter('use_sim_time', True)
        except ParameterAlreadyDeclaredException:
            pass
        if not self.get_parameter('use_sim_time').value:
            self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.get_logger().info("use_sim_time=True")

        self.declare_parameter('min_waypoint_separation_m', 1.0)
        self.declare_parameter('enable_los_shortcut', True)
        self.declare_parameter('auto_advance_on_proximity', False)
        self.declare_parameter('goal_reached_radius', 0.35)
        self.declare_parameter('goal_frame', 'auto')   

        self.obstacle_buffer_zone = 10
        self.allow_diagonal = True

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=False)

        self.map_info = None
        self.map_data_raw = None
        self.costmap_data = None

        self.robot_xy = None  
        self.path_cells = []
        self.waypoints = []
        self.total_wps = 0
        self.awaiting_ack = False
        self.last_sent_wp_grid = None

        goal_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self.pub_goal = self.create_publisher(PoseStamped, '/exploration_goal', goal_qos)

        map_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self.sub_map = self.create_subscription(OccupancyGrid, '/map', self.map_cb, map_qos)

        ok_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                            durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=10)
        self.sub_ok = self.create_subscription(Bool, '/goal_achieved', self.ok_cb, ok_qos)

        dest_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=10)
        self.sub_dest = self.create_subscription(PoseStamped, '/destination_goal', self.dest_goal_cb, dest_qos)
        self.get_logger().info("Subscribed to /destination_goal (PoseStamped).")

        self.create_timer(0.1, self.pose_tick)  
        self.create_timer(0.1, self.plot_tick)   

        plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))
        self.fig.suptitle('Costmap (click to set target)')
        self.ax.set_title('Costmap (inflated)')
        self.cmap_fields = ListedColormap(['gray', 'green', 'orange', 'red'])
        self.display_mapping_costmap = {-1: 0, 0: 1, 99: 2, 100: 3}
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        self.get_logger().info("PathPlannerNode ready. Click on the map to set a goal, or publish PoseStamped to /destination_goal.")

    def _plan_to_goal_cell(self, goal_c: int, goal_r: int):
        if self.robot_xy is None:
            self.get_logger().warn("Robot pose not available yet.")
            return

        if self.costmap_data is None or self.map_info is None:
            self.get_logger().warn("Map not available yet.")
            return

        if not (0 <= goal_c < self.map_info.width and 0 <= goal_r < self.map_info.height):
            self.get_logger().warn("Goal cell outside map bounds.")
            return

        start_c, start_r = self.world_to_grid(*self.robot_xy)
        goal_c, goal_r = self._nearest_free(goal_c, goal_r)
        if goal_c is None:
            self.get_logger().warn("No free goal near the requested point.")
            return

        path = self._astar((start_c, start_r), (goal_c, goal_r))
        if not path:
            self.get_logger().warn("No path found.")
            return

        self.path_cells = path
        #wps = self._simplify_to_corners(path)
        #if self.get_parameter('enable_los_shortcut').value:
            #wps = self._shortcut_waypoints(wps)
        #wps = self._space_waypoints_los(wps)
        wps = path
        self.waypoints = wps

        if wps and wps[0] == (start_c, start_r):
            wps = wps[1:]
        if wps:
            wx0, wy0 = self.grid_to_world(*wps[0])
            if math.hypot(wx0 - self.robot_xy[0], wy0 - self.robot_xy[1]) < 0.25:
                wps = wps[1:]
        if not wps:
            wps = [path[-1]]

        if not self._line_of_sight_free((start_c, start_r), wps[0]):
            for k in range(len(path) - 1, 0, -1):
                if self._line_of_sight_free((start_c, start_r), path[k]):
                    wps = [path[k]] + wps
                    break

        self.waypoints = wps
        self.total_wps = len(wps)
        self.last_sent_wp_grid = None
        self.awaiting_ack = False
        self.get_logger().info(f"Planned path with {len(path)} cells → {self.total_wps} waypoints.")
        self.publish_current_wp()

    def _nudge_out_of_buffer_if_needed(self):
        """
        If the robot's current grid cell is not free (i.e., in buffer 99 or obstacle 100),
        immediately publish a single escape waypoint to the nearest free cell (0).
        """
        if self.robot_xy is None or self.costmap_data is None or self.map_info is None:
            return

        rc, rr = self.world_to_grid(*self.robot_xy)
        if not (0 <= rr < self.costmap_data.shape[0] and 0 <= rc < self.costmap_data.shape[1]):
            return

        if self._is_free(rc, rr):
            return  

        esc_c, esc_r = self._nearest_free(rc, rr, max_radius=50)
        if esc_c is None:
            self.get_logger().warn("Robot ended in buffer/obstacle but no free cell found nearby.")
            return

        self.waypoints = [(esc_c, esc_r)]
        self.total_wps = 1
        self.last_sent_wp_grid = None
        self.awaiting_ack = False

        wx, wy = self.grid_to_world(esc_c, esc_r)
        self.get_logger().info(
            f"Robot ended inside buffer; nudging to nearest free cell at grid=({esc_c},{esc_r}) "
            f"map=({wx:.2f},{wy:.2f})."
        )
        self.publish_current_wp()

    def dest_goal_cb(self, msg: PoseStamped):
        """
        Accept a PoseStamped goal (any frame). Transform to 'map' if needed and plan like a click.
        """
        if self.map_info is None:
            self.get_logger().warn("Map not available yet; cannot process /destination_goal.")
            return

        try:
            if msg.header.frame_id and msg.header.frame_id != 'map':
                tf = self.tf_buffer.lookup_transform('map', msg.header.frame_id, Time(),
                                                     timeout=Duration(seconds=0.2))
                pose_in_map = do_transform_pose(msg.pose, tf)
            else:
                pose_in_map = msg.pose
        except TransformException as ex:
            self.get_logger().warn(f"TF transform of /destination_goal from '{msg.header.frame_id}' to 'map' failed: {ex}")
            return

        gx_m = pose_in_map.position.x
        gy_m = pose_in_map.position.y
        goal_c, goal_r = self.world_to_grid(gx_m, gy_m)
        self.get_logger().info(f"Received /destination_goal in map: ({gx_m:.2f}, {gy_m:.2f}) -> grid ({goal_c}, {goal_r})")

        self._plan_to_goal_cell(goal_c, goal_r)

    def map_cb(self, msg: OccupancyGrid):
        structure_changed = (
            self.map_info is None or
            msg.info.width != self.map_info.width or
            msg.info.height != self.map_info.height or
            msg.info.origin.position.x != self.map_info.origin.position.x or
            msg.info.origin.position.y != self.map_info.origin.position.y
        )
        self.map_info = msg.info
        self.map_data_raw = np.array(msg.data).reshape(msg.info.height, msg.info.width)

        obstacle_mask = (self.map_data_raw == 100)
        struct = np.ones((2 * self.obstacle_buffer_zone + 1, 2 * self.obstacle_buffer_zone + 1))
        inflated_mask = binary_dilation(obstacle_mask, structure=struct)

        cm = self.map_data_raw.copy()
        cm[inflated_mask] = 99
        cm[obstacle_mask] = 100
        self.costmap_data = cm

        if structure_changed:
            self.path_cells.clear()
            self.waypoints.clear()
            self.total_wps = 0
            self.last_sent_wp_grid = None
            self.awaiting_ack = False

    def on_click(self, event):
        if event.inaxes != self.ax or self.costmap_data is None or self.map_info is None:
            return
        if event.button != 1:
            return

        if self.robot_xy is None:
            self.get_logger().warn("Robot pose not available yet.")
            return

        goal_c = int(round(event.xdata)); goal_r = int(round(event.ydata))
        if not (0 <= goal_c < self.map_info.width and 0 <= goal_r < self.map_info.height):
            self.get_logger().warn("Clicked outside map bounds.")
            return

        start_c, start_r = self.world_to_grid(*self.robot_xy)
        goal_c, goal_r = self._nearest_free(goal_c, goal_r)
        if goal_c is None:
            self.get_logger().warn("No free goal near the clicked point.")
            return

        path = self._astar((start_c, start_r), (goal_c, goal_r))
        if not path:
            self.get_logger().warn("No path found.")
            return

        self.path_cells = path
        wps = self._simplify_to_corners(path)
        if self.get_parameter('enable_los_shortcut').value:
            wps = self._shortcut_waypoints(wps)
        wps = self._space_waypoints_los(wps)
        #wps = path                     
        self.waypoints = wps

        if wps and wps[0] == (start_c, start_r):
            wps = wps[1:]
        if wps:
            wx0, wy0 = self.grid_to_world(*wps[0])
            if math.hypot(wx0 - self.robot_xy[0], wy0 - self.robot_xy[1]) < 0.25:
                wps = wps[1:]
        if not wps:
            wps = [path[-1]]

        if not self._line_of_sight_free((start_c, start_r), wps[0]):
            for k in range(len(path) - 1, 0, -1):
                if self._line_of_sight_free((start_c, start_r), path[k]):
                    wps = [path[k]] + wps
                    break

        self.waypoints = wps
        self.total_wps = len(wps)
        self.last_sent_wp_grid = None
        self.awaiting_ack = False
        self.get_logger().info(f"Planned path with {len(path)} cells → {self.total_wps} waypoints.")
        self.publish_current_wp()

    def ok_cb(self, msg: Bool):
        self.get_logger().info(f"/goal_achieved: {msg.data} (awaiting_ack={self.awaiting_ack})")
        if msg.data and self.awaiting_ack:
            self.advance_wp()

    def advance_wp(self):
        if not self.waypoints:
            self.awaiting_ack = False
            return
        done = self.waypoints.pop(0)
        self.get_logger().info(f"Waypoint reached and removed: {done}")
        self.last_sent_wp_grid = None
        self.awaiting_ack = False
        if self.waypoints:
            self.publish_current_wp()
        else:
            self.get_logger().info("Path complete.")
            self._nudge_out_of_buffer_if_needed()

    def publish_current_wp(self):
        if not self.waypoints:
            self.get_logger().info("No more goals.")
            self.awaiting_ack = False
            return

        gx, gy = self.waypoints[0]
        if self.last_sent_wp_grid == (gx, gy):
            self.awaiting_ack = True
            return
        self.last_sent_wp_grid = (gx, gy)

        wx_m, wy_m = self.grid_to_world(gx, gy)
        if len(self.waypoints) >= 2:
            nx_m, ny_m = self.grid_to_world(*self.waypoints[1])
            yaw_m = math.atan2(ny_m - wy_m, nx_m - wx_m)
        else:
            yaw_m = 0.0

        ps_map = PoseStamped()
        ps_map.header.stamp = self.get_clock().now().to_msg()
        ps_map.header.frame_id = 'map'
        ps_map.pose.position.x = wx_m
        ps_map.pose.position.y = wy_m
        ps_map.pose.position.z = 0.0
        ps_map.pose.orientation = yaw_to_quaternion(yaw_m)

        target = self.get_parameter('goal_frame').get_parameter_value().string_value or 'auto'
        frame_used = 'map'
        out = ps_map

        try:
            if target == 'map':
                pass
            elif target == 'odom' or (target == 'auto' and self.tf_buffer.can_transform('odom', 'map', Time())):
                tf = self.tf_buffer.lookup_transform('odom', 'map', Time(), timeout=Duration(seconds=0.1))
                pose = do_transform_pose(ps_map.pose, tf)
                out = PoseStamped()
                out.header.stamp = ps_map.header.stamp
                out.header.frame_id = 'odom'
                out.pose = pose
                frame_used = 'odom'
            elif target == 'base_link':
                tf = self.tf_buffer.lookup_transform('base_link', 'map', Time(), timeout=Duration(seconds=0.1))
                pose = do_transform_pose(ps_map.pose, tf)
                out = PoseStamped()
                out.header.stamp = ps_map.header.stamp
                out.header.frame_id = 'base_link'
                out.pose = pose
                frame_used = 'base_link'
            else:
                pass
        except TransformException as ex:
            self.get_logger().warn(f"TF transform to '{target}' failed ({ex}), publishing in 'map'.")

        sent_idx = self.total_wps - len(self.waypoints) + 1
        self.pub_goal.publish(out)
        self.awaiting_ack = True
        self.get_logger().info(
            f"Published waypoint {sent_idx}/{self.total_wps} in {frame_used}: "
            f"map=({wx_m:.2f},{wy_m:.2f})"
            + ("" if frame_used == 'map' else f" -> {frame_used}=({out.pose.position.x:.2f},{out.pose.position.y:.2f})")
        )

    def pose_tick(self):
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_link', Time(), timeout=Duration(seconds=0.05))
            self.robot_xy = (tf.transform.translation.x, tf.transform.translation.y)
        except TransformException:
            self.robot_xy = None

        if self.get_parameter('auto_advance_on_proximity').value and self.waypoints and self.robot_xy:
            wx, wy = self.grid_to_world(*self.waypoints[0])
            if math.hypot(self.robot_xy[0] - wx, self.robot_xy[1] - wy) < float(self.get_parameter('goal_reached_radius').value):
                self.advance_wp()

    def plot_tick(self):
        if self.costmap_data is None or self.map_info is None:
            return
        self.ax.clear()
        disp = np.vectorize(self.display_mapping_costmap.get)(self.costmap_data)
        self.ax.imshow(disp, cmap=self.cmap_fields, vmin=0, vmax=len(self.display_mapping_costmap)-1, origin='lower')
        self.ax.set_title(f'Mapa (promien strefy {self.obstacle_buffer_zone})')
        self.ax.set_xlabel('X '); self.ax.set_ylabel('Y ')

        if self.waypoints:
            if self.robot_xy:
                rc, rr = self.world_to_grid(*self.robot_xy)
                w0c, w0r = self.waypoints[0]
                self.ax.plot([rc, w0c], [rr, w0r], '-', linewidth=2, label='Trasa')
            for i in range(len(self.waypoints)-1):
                c0, r0 = self.waypoints[i]; c1, r1 = self.waypoints[i+1]
                self.ax.plot([c0, c1], [r0, r1], '-', linewidth=2)
            wx = [w[0] for w in self.waypoints]; wy = [w[1] for w in self.waypoints]
            self.ax.plot(wx, wy, 'b*', markersize=10, label='pozostale punkty')
            self.ax.plot(self.waypoints[0][0], self.waypoints[0][1], 'r*', markersize=14, label='Obecny cel')

        if self.robot_xy:
            gx, gy = self.world_to_grid(*self.robot_xy)
            self.ax.plot(gx, gy, 'ro', markersize=6, label='Robot')

        if self.ax.get_legend_handles_labels()[1]:
            self.ax.legend(loc='upper right')
        plt.draw(); plt.pause(0.001)

    def on_motion(self, event):
        if event.inaxes == self.ax and self.map_info and event.xdata is not None and event.ydata is not None:
            col = int(event.xdata); row = int(event.ydata)
            if 0 <= col < self.map_info.width and 0 <= row < self.map_info.height:
                mx, my = self.grid_to_world(col, row)
                self.fig.suptitle(f'Cursor: [row: {row}, col: {col}] | Map (x: {mx:.2f}m, y: {my:.2f}m)')

    def grid_to_world(self, c, r):
        x = self.map_info.origin.position.x + (c + 0.5) * self.map_info.resolution
        y = self.map_info.origin.position.y + (r + 0.5) * self.map_info.resolution
        return x, y

    def world_to_grid(self, x, y):
        c = int((x - self.map_info.origin.position.x) / self.map_info.resolution)
        r = int((y - self.map_info.origin.position.y) / self.map_info.resolution)
        return c, r

    def _is_free(self, c, r):
        return (0 <= r < self.costmap_data.shape[0] and
                0 <= c < self.costmap_data.shape[1] and
                self.costmap_data[r, c] == 0)

    def _nearest_free(self, c, r, max_radius=20):
        if self._is_free(c, r): return c, r
        for rad in range(1, max_radius+1):
            for dc in range(-rad, rad+1):
                for dr in (-rad, rad):
                    cc, rr = c+dc, r+dr
                    if self._is_free(cc, rr): return cc, rr
            for dr in range(-rad+1, rad):
                for dc in (-rad, rad):
                    cc, rr = c+dc, r+dr
                    if self._is_free(cc, rr): return cc, rr
        return None, None

    def _neighbors(self, c, r):
        if self.allow_diagonal:
            deltas = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        else:
            deltas = [(-1,0),(1,0),(0,-1),(0,1)]
        for dc, dr in deltas:
            nc, nr = c+dc, r+dr
            if self._is_free(nc, nr):
                if dc!=0 and dr!=0 and not (self._is_free(c+dc, r) and self._is_free(c, r+dr)):
                    continue
                yield (nc, nr), (math.sqrt(2.0) if dc and dr else 1.0)

    def _astar(self, start, goal):
        sc, sr = start; gc, gr = goal
        if not self._is_free(sc, sr) or not self._is_free(gc, gr): return []
        open_set = [(0.0, start)]
        came = {}; g = {start: 0.0}
        def h(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])
        closed = set()
        while open_set:
            _, cur = heapq.heappop(open_set)
            if cur in closed: continue
            if cur == goal:
                path = [cur]
                while cur in came:
                    cur = came[cur]; path.append(cur)
                path.reverse(); return path
            closed.add(cur)
            for (nbr, cost) in self._neighbors(*cur):
                t = g[cur] + cost
                if nbr in g and t >= g[nbr]: continue
                came[nbr] = cur; g[nbr] = t
                heapq.heappush(open_set, (t + h(nbr, goal), nbr))
        return []

    def _bresenham_cells(self, c0, r0, c1, r1):
        dc = abs(c1-c0); dr = abs(r1-r0)
        sc = 1 if c0 < c1 else -1; sr = 1 if r0 < r1 else -1
        err = dc - dr; c, r = c0, r0
        while True:
            yield c, r
            if c==c1 and r==r1: break
            e2 = 2*err
            if e2 > -dr: err -= dr; c += sc
            if e2 <  dc: err += dc; r += sr

    def _line_of_sight_free(self, a, b):
        for c, r in self._bresenham_cells(a[0], a[1], b[0], b[1]):
            if not self._is_free(c, r): return False
        return True

    def _simplify_to_corners(self, path_cells):
        if len(path_cells) <= 2: return path_cells[:]
        out = [path_cells[0]]
        prev, curr = path_cells[0], path_cells[1]
        def nrm(d): dx,dy=d; return (0 if dx==0 else int(dx/abs(dx)), 0 if dy==0 else int(dy/abs(dy)))
        prev_dir = (curr[0]-prev[0], curr[1]-prev[1])
        for i in range(2, len(path_cells)):
            nxt = path_cells[i]
            cur_dir = (nxt[0]-curr[0], nxt[1]-curr[1])
            if nrm(cur_dir) != nrm(prev_dir): out.append(curr)
            prev, curr, prev_dir = curr, nxt, cur_dir
        out.append(path_cells[-1]); return out

    def _shortcut_waypoints(self, wps):
        if len(wps) <= 2: return wps[:]
        out = []; i = 0
        while i < len(wps)-1:
            j = len(wps)-1
            while j > i+1 and not self._line_of_sight_free(wps[i], wps[j]): j -= 1
            out.append(wps[i]); i = j
        out.append(wps[-1]); return out

    def _space_waypoints_los(self, wps):
        if not wps: return []
        min_sep = float(self.get_parameter('min_waypoint_separation_m').value)
        out = [wps[0]]; i = 0
        while i < len(wps)-1:
            j = i+1; chosen = j
            while j < len(wps) and self._line_of_sight_free(wps[i], wps[j]):
                wxi, wyi = self.grid_to_world(*wps[i]); wxj, wyj = self.grid_to_world(*wps[j])
                d = math.hypot(wxj-wxi, wyj-wyi); chosen = j
                if d >= min_sep: break
                j += 1
            out.append(wps[chosen]); i = chosen
        # dedupe
        out2 = [out[0]]
        for k in range(1, len(out)):
            if out[k] != out2[-1]: out2.append(out[k])
        return out2


def main():
    try:
        rclpy.init()
        node = PathPlannerNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"[path_planner] Fatal error: {e}")
        raise
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()
        try:
            plt.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()