#!/usr/bin/env python3
import os
import math
import heapq
import numpy as np

# ---- FORCE A GUI BACKEND BEFORE importing pyplot (so the window opens) ----
import matplotlib
try:
    matplotlib.use("Qt5Agg")      # needs python3-pyqt5
except Exception:
    try:
        matplotlib.use("TkAgg")   # needs python3-tk
    except Exception:
        pass  # fall back; if Agg, window won't show (environment issue)

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
from sensor_msgs.msg import Range, LaserScan
from std_srvs.srv import Empty

import tf2_ros
from tf2_ros import TransformException
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose

# Optional BT viz
_HAVE_BT = True
try:
    import py_trees
    import py_trees_ros
except Exception:
    _HAVE_BT = False


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw / 2.0); q.z = math.sin(yaw / 2.0)
    q.x = 0.0; q.y = 0.0
    return q


class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')

        # ---------- params ----------
        try:
            self.declare_parameter('use_sim_time', True)    #False
        except ParameterAlreadyDeclaredException:
            pass
        if not self.get_parameter('use_sim_time').value:
            # change to True/False to match your system if needed
            self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.get_logger().info("use_sim_time=True")

        self.declare_parameter('min_waypoint_separation_m', 1.0)
        self.declare_parameter('enable_los_shortcut', True)
        self.declare_parameter('auto_advance_on_proximity', False)
        self.declare_parameter('goal_reached_radius', 0.35)
        self.declare_parameter('goal_frame', 'auto')   # 'auto'|'map'|'odom'|'base_link'

        # NEW safety-stop params
        self.declare_parameter('safety_stop_enable', True)
        self.declare_parameter('stop_distance_m', 0.10)
        self.declare_parameter('range_topics', [])     # e.g. ['front_range','rear_range']
        self.declare_parameter('scan_topics', [])      # e.g. ['/scan']
        self.declare_parameter('enable_bt_viz', True)

        self.safety_stop_enable: bool = self.get_parameter('safety_stop_enable').value
        self.stop_distance_m: float = float(self.get_parameter('stop_distance_m').value)
        self.bt_viz_enabled: bool = bool(self.get_parameter('enable_bt_viz').value)

        self.obstacle_buffer_zone = 5
        self.allow_diagonal = True

        # ---------- TF ----------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=False)

        # ---------- state ----------
        self.map_info = None
        self.map_data_raw = None
        self.costmap_data = None

        self.robot_xy = None  # (x,y) in map frame
        self.path_cells = []
        self.waypoints = []
        self.total_waypoints = 0
        self.awaiting_ack = False
        self.last_sent_wp_grid = None

        # queue a goal if we must escape buffer first
        self.pending_goal_cell = None  # (c, r) or None

        # Safety stop state
        self.safety_stop_active = False
        self.min_obstacle_distance = None

        # ---------- pubs/subs ----------
        goal_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self.pub_goal = self.create_publisher(PoseStamped, '/exploration_goal', goal_qos)

        self.pub_safety = self.create_publisher(Bool, '/safety_stop', 1)

        map_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self.sub_map = self.create_subscription(OccupancyGrid, '/map', self.map_cb, map_qos)

        ok_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                            durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=10)
        self.sub_ok = self.create_subscription(Bool, '/goal_achieved', self.ok_cb, ok_qos)

        # destination goal
        dest_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=10)
        self.sub_dest = self.create_subscription(PoseStamped, '/destination_goal', self.dest_goal_cb, dest_qos)
        self.get_logger().info("Subscribed to /destination_goal (PoseStamped).")

        # Range & Scan subscriptions (configurable lists)
        self._range_subs = []
        self._scan_subs = []
        for topic in list(self.get_parameter('range_topics').value or []):
            self._range_subs.append(
                self.create_subscription(Range, topic, self._range_cb, 10)
            )
            self.get_logger().info(f"Safety: subscribed to Range topic: {topic}")
        for topic in list(self.get_parameter('scan_topics').value or []):
            self._scan_subs.append(
                self.create_subscription(LaserScan, topic, self._scan_cb, 10)
            )
            self.get_logger().info(f"Safety: subscribed to LaserScan topic: {topic}")

        # reset
        self.srv_clear = self.create_service(Empty, '/clear_safety_stop', self._srv_clear_safety_stop)

        # timers
        self.create_timer(0.1, self.pose_tick)
        self.create_timer(0.1, self.plot_tick)

        # rysownie
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))
        self.fig.suptitle('Wybierz cel')
        self.ax.set_title('mapa')
        self.cmap_fields = ListedColormap(['gray', 'green', 'orange', 'red'])
        self.display_mapping_costmap = {-1: 0, 0: 1, 99: 2, 100: 3}
        self.fig.canvas.mpl_connect('ruch', self.on_motion)
        self.fig.canvas.mpl_connect('przycisk', self.on_click)

        # bt
        self.bt_tree = None
        if _HAVE_BT and self.bt_viz_enabled:
            try:
                self._bt_setup()
                self.get_logger().info("BT viz running on node name: /pp_bt_tree")
            except Exception as e:
                self.get_logger().warn(f"BT viz failed to start: {e}")
        elif not _HAVE_BT and self.bt_viz_enabled:
            self.get_logger().warn("py_trees/py_trees_ros not installed; BT viz disabled.")

        self.get_logger().info("PathPlannerNode ready. Click on the map to set a goal, or publish PoseStamped to /destination_goal.")

    # ==================== SAFETY STOP ====================
    def _range_cb(self, msg: Range):
        if not self.safety_stop_enable:
            return
        if np.isfinite(msg.range) and msg.range >= 0.0:
            self._update_min_distance(msg.range)

    def _scan_cb(self, msg: LaserScan):
        if not self.safety_stop_enable:
            return
        vals = [r for r in msg.ranges if np.isfinite(r) and r >= 0.0]
        if not vals:
            return
        self._update_min_distance(min(vals))

    def _update_min_distance(self, d: float):
        # Keep a rolling min for info only (no filtering necessary here)
        self.min_obstacle_distance = d
        if not self.safety_stop_active and d <= self.stop_distance_m:
            self._trigger_safety_stop(d)

    def _trigger_safety_stop(self, d: float):
        self.safety_stop_active = True
        self.waypoints.clear()
        self.awaiting_ack = False
        self.last_sent_wp_grid = None
        # publish latched-like safety flag (depth=1 is enough here)
        msg = Bool(); msg.data = True
        self.pub_safety.publish(msg)
        self.get_logger().warn(f"[SAFETY STOP] Min distance {d:.3f} m <= {self.stop_distance_m:.3f} m. Stopping planner, clearing waypoints.")
        # From now on, publish_point / plan requests will be ignored until cleared

    def _srv_clear_safety_stop(self, req, res):
        self.safety_stop_active = False
        self.get_logger().info("Safety stop cleared via service /clear_safety_stop.")
        msg = Bool(); msg.data = False
        self.pub_safety.publish(msg)
        return res

    # ==================== BT VIZ ====================
    def _bt_setup(self):
        # Behaviours
        class SafeDistanceBT(py_trees.behaviour.Behaviour):
            def __init__(self, outer):
                super().__init__(name="SafeDistance")
                self.outer = outer
            def update(self):
                if self.outer.safety_stop_active:
                    return py_trees.common.Status.FAILURE
                return py_trees.common.Status.SUCCESS

        class PlannerActiveBT(py_trees.behaviour.Behaviour):
            def __init__(self, outer):
                super().__init__(name="PlannerActive")
                self.outer = outer
            def update(self):
                return py_trees.common.Status.SUCCESS if len(self.outer.waypoints) > 0 else py_trees.common.Status.FAILURE

        root = py_trees.composites.Parallel(name="PP_Root", policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False))
        root.add_child(SafeDistanceBT(self))
        root.add_child(PlannerActiveBT(self))

        self.bt_tree = py_trees_ros.trees.BehaviourTree(root)
        # expose services under /pp_bt_tree
        self.bt_tree.setup(node=self, node_name="pp_bt_tree")
        # tick regularly
        self.bt_tree.tick_tock(period_ms=300)

    # ==================== PLANNING CORE ====================
    def _plan_to_goal_cell(self, goal_c: int, goal_r: int):
        if self.safety_stop_active:
            self.get_logger().warn("stop - przeszkoda w zasiegu")
            return

        if self.robot_xy is None:
            self.get_logger().warn("brak pozyji robota - amcl")
            return

        if self.costmap_data is None or self.map_info is None:
            self.get_logger().warn("nie otrzymano mapy")
            return

        if not (0 <= goal_c < self.map_info.width and 0 <= goal_r < self.map_info.height):
            self.get_logger().warn("cel poza mapa")
            return

        start_c, start_r = self.world_to_grid(*self.robot_xy)

        if not self._is_free(start_c, start_r):
            esc_c, esc_r = self._nearest_free(start_c, start_r, max_radius=50)
            if esc_c is None:
                self.get_logger().warn("Robot is in buffer/obstacle but no free start cell found nearby.")
                return
            self.pending_goal_cell = (goal_c, goal_r)
            self.get_logger().info(f"Start not free; escaping to ({esc_c},{esc_r}) then planning to {self.pending_goal_cell}.")
            self.waypoints = [(esc_c, esc_r)]
            self.total_waypoints = 1
            self.last_sent_wp_grid = None
            self.awaiting_ack = False
            self.publish_point()
            return

        # here screen
        goal_c, goal_r = self._nearest_free(goal_c, goal_r)
        if goal_c is None:
            self.get_logger().warn("brak wolnej komórki w pobliżu celu.")
            return

        path = self._astar((start_c, start_r), (goal_c, goal_r))
        if not path:
            self.get_logger().warn("brak sciezki.")
            return

        self.path_cells = path
        waypoints = self._simplify_to_corners(path)
        if self.get_parameter('enable_los_shortcut').value:
            waypoints = self._shortcut_waypoints(waypoints)
        waypoints = self._space_waypoints_los(waypoints)

        if waypoints and waypoints[0] == (start_c, start_r):
            waypoints = waypoints[1:]
        if waypoints:
            wx0, wy0 = self.grid_to_world(*waypoints[0])
            if math.hypot(wx0 - self.robot_xy[0], wy0 - self.robot_xy[1]) < 0.25:
                waypoints = waypoints[1:]
        if not waypoints:
            waypoints = [path[-1]]

        if not self._line_of_sight_free((start_c, start_r), waypoints[0]):
            for k in range(len(path) - 1, 0, -1):
                if self._line_of_sight_free((start_c, start_r), path[k]):
                    waypoints = [path[k]] + waypoints
                    break

        self.waypoints = waypoints
        self.total_waypoints = len(waypoints)
        self.last_sent_wp_grid = None
        self.awaiting_ack = False
        self.get_logger().info(f"sciezka z {len(path)} komorek → {self.total_waypoints} punktow")
        self.publish_point()

    # ---------- buffer escape helpers ----------
    def _nudge_out_of_buffer_if_needed(self):
        if self.robot_xy is None or self.costmap_data is None or self.map_info is None:
            return
        c, r = self.world_to_grid(*self.robot_xy)
        if not (0 <= r < self.costmap_data.shape[0] and 0 <= c < self.costmap_data.shape[1]):
            return
        if self._is_free(c, r):
            return
        esc_c, esc_r = self._nearest_free(c, r, max_radius=50)
        if esc_c is None:
            self.get_logger().warn("Robot ended in buffer/obstacle but no free cell found nearby.")
            return
        self.waypoints = [(esc_c, esc_r)]
        self.total_waypoints = 1
        self.last_sent_wp_grid = None
        self.awaiting_ack = False
        wx, wy = self.grid_to_world(esc_c, esc_r)
        self.get_logger().info(
            f"Robot inside buffer; nudging to nearest free cell at grid=({esc_c},{esc_r}) map=({wx:.2f},{wy:.2f})."
        )
        self.publish_point()

    # ---------- /destination_goal ----------
    def dest_goal_cb(self, msg: PoseStamped):
        if self.safety_stop_active:
            self.get_logger().warn("Safety stop active — ignoring /destination_goal.")
            return

        if self.map_info is None:
            self.get_logger().warn("Map not available yet; cannot process /destination_goal.")
            return

        # Transform to map frame if necessary
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

        # If currently inside buffer, escape first and queue this goal
        if self.robot_xy is not None:
            sc, sr = self.world_to_grid(*self.robot_xy)
            if not self._is_free(sc, sr):
                esc_c, esc_r = self._nearest_free(sc, sr, max_radius=50)
                if esc_c is not None:
                    self.pending_goal_cell = (goal_c, goal_r)
                    self.get_logger().info(f"In buffer now; escape to ({esc_c},{esc_r}) then plan to {self.pending_goal_cell}.")
                    self.waypoints = [(esc_c, esc_r)]
                    self.total_waypoints = 1
                    self.last_sent_wp_grid = None
                    self.awaiting_ack = False
                    self.publish_point()
                    return

        self._plan_to_goal_cell(goal_c, goal_r)

    # mapa
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
            self.total_waypoints = 0
            self.last_sent_wp_grid = None
            self.awaiting_ack = False
            

    # ---------- click to set target ----------
    def on_click(self, event):
        if event.inaxes != self.ax or self.costmap_data is None or self.map_info is None:
            return
        if event.button != 1:
            return
        if self.safety_stop_active:
            self.get_logger().warn("stop")
            return

        if self.robot_xy is None:
            self.get_logger().warn("brak pozycji - amcl")
            return

        goal_c = int(round(event.xdata))
        goal_r = int(round(event.ydata))
        if not (0 <= goal_c < self.map_info.width and 0 <= goal_r < self.map_info.height):
            self.get_logger().warn("wspolrzedne poza mapa")
            return

        sc, sr = self.world_to_grid(*self.robot_xy)
        if not self._is_free(sc, sr):
            esc_c, esc_r = self._nearest_free(sc, sr, max_radius=50)
            if esc_c is not None:
                self.pending_goal_cell = (goal_c, goal_r)
                self.get_logger().info(f"Click while in buffer; escape to ({esc_c},{esc_r}) then plan to {self.pending_goal_cell}.")
                self.waypoints = [(esc_c, esc_r)]
                self.total_waypoints = 1
                self.last_sent_wp_grid = None
                self.awaiting_ack = False
                self.publish_point()
                return

        goal_c, goal_r = self._nearest_free(goal_c, goal_r)
        if goal_c is None:
            self.get_logger().warn("pozycja zajeta")
            return

        self._plan_to_goal_cell(goal_c, goal_r)

    # ---------- progression ----------
    def ok_cb(self, msg: Bool):
        self.get_logger().info(f"/goal_achieved: {msg.data} (awaiting_ack={self.awaiting_ack})")
        if not msg.data:
            return

        if self.awaiting_ack:
            self.advance_wp()
        else:
            # even if we weren't waiting, escape if we're in buffer
            self._nudge_out_of_buffer_if_needed()
            if not self.waypoints and self.pending_goal_cell is not None:
                gc, gr = self.pending_goal_cell
                self.pending_goal_cell = None
                self._plan_to_goal_cell(gc, gr)

    def advance_wp(self):
        if not self.waypoints:
            self.awaiting_ack = False
            return
        done = self.waypoints.pop(0)
        self.get_logger().info(f"Osiagnieto cel {done}")
        self.last_sent_wp_grid = None
        self.awaiting_ack = False

        if self.waypoints:
            self.publish_point()
        else:
            self.get_logger().info("Sciezka zakonczona")
            self._nudge_out_of_buffer_if_needed()
            if not self.waypoints and self.pending_goal_cell is not None:
                gc, gr = self.pending_goal_cell
                self.pending_goal_cell = None
                self._plan_to_goal_cell(gc, gr)

    def publish_point(self):
        if not self.waypoints:
            self.get_logger().info("koniec punktow")
            self.awaiting_ack = False
            return
        if self.safety_stop_active:
            self.get_logger().warn("zatrzymanie - przeszkoda")
            self.awaiting_ack = False
            return

        gx, gy = self.waypoints[0]
        if self.last_sent_wp_grid == (gx, gy):
            self.awaiting_ack = True
            return
        self.last_sent_wp_grid = (gx, gy)

        # kordy
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

        # parametrem goal_frame wybierasz w jakim ukladzie wybierasz cel map/odom/base_link
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
            self.get_logger().warn(f"nie znalezono tf '{target}' do ({ex})")

        sent_idx = self.total_waypoints - len(self.waypoints) + 1
        self.pub_goal.publish(out)
        self.awaiting_ack = True
        self.get_logger().info(
            f"nadano punkt {sent_idx}/{self.total_waypoints} w {frame_used}: "
            f"map=({wx_m:.2f},{wy_m:.2f})"
            + ("" if frame_used == 'map' else f" -> {frame_used}=({out.pose.position.x:.2f},{out.pose.position.y:.2f})")
        )

    # ---------- pose/plot ----------
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
        self.ax.set_title(f'Mapa {self.obstacle_buffer_zone} promien strefy)')
        self.ax.set_xlabel('X '); self.ax.set_ylabel('Y ')

        # draw straight segments robot actually follows
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

        # overlay current min distance + safety status
        #status = "SAFE" if not self.safety_stop_active else "STOPPED"
        #md = "n/a" if self.min_obstacle_distance is None else f"{self.min_obstacle_distance:.2f} m"
        #self.ax.text(0.02, 0.98, f"Safety: {status}\nmin d: {md}\nthr: {self.stop_distance_m:.2f} m",
                     #transform=self.ax.transAxes, va='top', ha='left', fontsize=9,
                    # bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        if self.ax.get_legend_handles_labels()[1]:
            self.ax.legend(loc='upper right')
        plt.draw(); plt.pause(0.001)

    # ---------- utils / helpers ----------
    def on_motion(self, event):
        if event.inaxes == self.ax and self.map_info and event.xdata is not None and event.ydata is not None:
            col = int(event.xdata); row = int(event.ydata)
            if 0 <= col < self.map_info.width and 0 <= row < self.map_info.height:
                mx, my = self.grid_to_world(col, row)
                self.fig.suptitle(f'Punkt: [x: {row}, y: {col}] | Mapa (x: {mx:.2f}m, x: {my:.2f}m)')

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
        sc, sr = start
        gc, gr = goal


        if not self._is_free(sc, sr) or not self._is_free(gc, gr):
            return []

        openqueue = [(0.0, start)]      # sterta z ktorej wybieram najlepsza komorke
        gcost = {start: 0.0}            # koszt dojscia do wezla
        poprzednik = {}                 
        closedset = set()               

        def heuristic(a, b):
            # przeciwprostokatna
            return math.hypot(a[0] - b[0], a[1] - b[1])

        while openqueue:
            # komorka o najmniejszym f = g + h
            _, current = heapq.heappop(openqueue)
            if current in closedset:
                continue

            if current == goal:
                sciezka = [current]
                while current in poprzednik:
                    current = poprzednik[current]
                    sciezka.append(current)
                sciezka.reverse()
                return sciezka

            closedset.add(current)

            for (neighbor, koszt) in self._neighbors(*current):
                tentative = gcost[current] + koszt  # g'(neighbor) = g(current) + c(current,neighbor)

                if neighbor in gcost and tentative >= gcost[neighbor]:
                    continue

                poprzednik[neighbor] = current
                gcost[neighbor] = tentative
                fscore = tentative + heuristic(neighbor, goal)
                heapq.heappush(openqueue, (fscore, neighbor))

        return []


    def _bresenham_cells(self, c0, r0, c1, r1):
        dc = abs(c1-c0); dr = abs(r1-r0)
        sc = 1 if c0 < c1 else -1
        sr = 1 if r0 < r1 else -1
        err = dc - dr; c, r = c0, r0
        while True:
            yield c, r
            if c==c1 and r==r1: break
            e2 = 2*err
            if e2 > -dr: 
                err -= dr; 
                c += sc
            if e2 <  dc:
                 err += dc; 
                 r += sr

    def _line_of_sight_free(self, a, b):
        for c, r in self._bresenham_cells(a[0], a[1], b[0], b[1]):
            if not self._is_free(c, r): return False
        return True

    def _simplify_to_corners(self, path_cells):
        if len(path_cells) <= 2: return path_cells[:]
        out = [path_cells[0]]
        prev, curr = path_cells[0], path_cells[1]
        def normalize(d): dx,dy=d; return (0 if dx==0 else int(dx/abs(dx)), 0 if dy==0 else int(dy/abs(dy)))
        prev_dir = (curr[0]-prev[0], curr[1]-prev[1])
        for i in range(2, len(path_cells)):
            next_punkt = path_cells[i]
            cur_dir = (next_punkt[0]-curr[0], next_punkt[1]-curr[1])
            if normalize(cur_dir) != normalize(prev_dir): out.append(curr)
            prev, curr, prev_dir = curr, next_punkt, cur_dir
        out.append(path_cells[-1]); return out

    def _shortcut_waypoints(self, waypoints):
        if len(waypoints) <= 2: return waypoints[:]
        out = []; i = 0
        while i < len(waypoints)-1:
            j = len(waypoints)-1
            while j > i+1 and not self._line_of_sight_free(waypoints[i], waypoints[j]): j -= 1
            out.append(waypoints[i]); i = j
        out.append(waypoints[-1]); return out

    def _space_waypoints_los(self, waypoints):
        if not waypoints: return []
        min_sep = float(self.get_parameter('min_waypoint_separation_m').value)
        out = [waypoints[0]]; i = 0
        while i < len(waypoints)-1:
            j = i+1; chosen = j
            while j < len(waypoints) and self._line_of_sight_free(waypoints[i], waypoints[j]):
                wxi, wyi = self.grid_to_world(*waypoints[i]); wxj, wyj = self.grid_to_world(*waypoints[j])
                d = math.hypot(wxj-wxi, wyj-wyi); chosen = j
                if d >= min_sep: break
                j += 1
            out.append(waypoints[chosen]); i = chosen
        # dodupy
        out2 = [out[0]]
        for k in range(1, len(out)):
            if out[k] != out2[-1]: out2.append(out[k])
        return out2


def main():
    node = None
    try:
        rclpy.init()
        node = PathPlannerNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"[planer] error: {e}")
        raise
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node is not None:
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
