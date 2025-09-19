#!/usr/bin/env python3
"""
Go-To-Goal controller as a Behaviour Tree.

Sub:
  /exploration_goal            (geometry_msgs/PoseStamped)
  /rosbot_base_controller/odom (nav_msgs/Odometry)  [optional if driving in 'map']
Pub:
  /cmd_vel                     (geometry_msgs/Twist)
  /goal_achieved               (std_msgs/Bool)  [TRANSIENT_LOCAL]
"""

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.parameter import Parameter
from rclpy.exceptions import ParameterAlreadyDeclaredException

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool

from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose

import py_trees
import py_trees_ros
import py_trees.common


def yaw_from_quaternion(q):
    qw, qx, qy, qz = q.w, q.x, q.y, q.z
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

def angle_wrap(a):
    while a > math.pi: a -= 2.0*math.pi
    while a < -math.pi: a += 2.0*math.pi
    return a


class GoToGoalNode(Node):
    def __init__(self):
        super().__init__('go_to_exploration_goal_bt')

        # --- use_sim_time: safe declare & ensure True ---
        try:
            self.declare_parameter('use_sim_time', True)
        except ParameterAlreadyDeclaredException:
            pass
        p = self.get_parameter('use_sim_time')
        if p.type_ == Parameter.Type.NOT_SET or (p.type_ == Parameter.Type.BOOL and not p.value):
            self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        # Parameters
        self.declare_parameters('', [
            ('working_frame', 'map'),         # <<< default: drive in MAP so numbers match the GUI
            ('linear_speed_limit', 0.35),
            ('angular_speed_limit', 1.2),
            ('linear_accel_limit', 0.6),
            ('angular_accel_limit', 2.0),
            ('k_rho', 1.0),
            ('k_alpha', 2.5),
            ('k_final_yaw', 1.0),
            ('goal_pos_tolerance', 0.15),
            ('goal_yaw_tolerance', 0.2),
            ('assume_map_equals_odom_if_no_tf', True),
        ])

        # Gains/limits
        self.v_max  = float(self.get_parameter('linear_speed_limit').value)
        self.w_max  = float(self.get_parameter('angular_speed_limit').value)
        self.a_v    = float(self.get_parameter('linear_accel_limit').value)
        self.a_w    = float(self.get_parameter('angular_accel_limit').value)
        self.k_rho  = float(self.get_parameter('k_rho').value)
        self.k_alpha= float(self.get_parameter('k_alpha').value)
        self.k_yaw  = float(self.get_parameter('k_final_yaw').value)
        self.pos_tol= float(self.get_parameter('goal_pos_tolerance').value)
        self.yaw_tol= float(self.get_parameter('goal_yaw_tolerance').value)
        self.assume_equal = bool(self.get_parameter('assume_map_equals_odom_if_no_tf').value)
        self.working_frame = self.get_parameter('working_frame').get_parameter_value().string_value or 'map'

        # TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # pubs
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_goal_achieved = self.create_publisher(
            Bool, '/goal_achieved',
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST, depth=1,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL)
        )
        self._goal_announced = False

        # subs
        # We still subscribe to odom for those who want to drive in odom.
        self.sub_odom = self.create_subscription(Odometry, '/rosbot_base_controller/odom', self.odom_cb, 10)
        qos_goal = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST, depth=1,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.sub_goal = self.create_subscription(PoseStamped, '/exploration_goal', self.goal_cb, qos_goal)

        # state
        self.current_pose: Optional[tuple] = None   # (x,y,yaw) in working_frame
        self.goal_msg: Optional[PoseStamped] = None
        self.goal_reached = True
        self.v_prev = 0.0; self.w_prev = 0.0

        # timer: keep pose fresh from TF when driving in 'map'
        self.create_timer(0.05, self._pose_tick_tf)

        self.get_logger().info(f'BT node up. working_frame="{self.working_frame}". Waiting for goals...')

    # ---- callbacks / pose sources ----
    def odom_cb(self, msg: Odometry):
        # Used if working_frame=='odom'
        if self.working_frame != 'odom':
            return
        p = msg.pose.pose.position; q = msg.pose.pose.orientation
        self.current_pose = (p.x, p.y, yaw_from_quaternion(q))

    def _pose_tick_tf(self):
        # If working_frame is MAP (or anything != odom), read pose via TF each tick
        if self.working_frame == 'odom':
            return
        try:
            tf = self.tf_buffer.lookup_transform(self.working_frame, 'base_link', Time(), timeout=Duration(seconds=0.05))
            x = tf.transform.translation.x
            y = tf.transform.translation.y
            z = tf.transform.rotation.z
            w = tf.transform.rotation.w
            yaw = math.atan2(2.0*w*z, 1.0 - 2.0*(z*z))
            self.current_pose = (x, y, yaw)
        except (LookupException, ConnectivityException, ExtrapolationException):
            self.current_pose = None

    def goal_cb(self, msg: PoseStamped):
        self.goal_msg = msg; self.goal_reached = False; self._goal_announced = False
        self.get_logger().info(f"New goal: frame='{msg.header.frame_id or '?'}' x={msg.pose.position.x:.3f} y={msg.pose.position.y:.3f}")

    # ---- helpers ----
    def stop(self):
        self.publish_cmd(0.0, 0.0)

    def publish_cmd(self, v, w):
        # accel limiting
        dt = 0.05
        def limit(prev, target, amax):
            dv = target - prev; step = amax * dt
            if dv >  step: target = prev + step
            if dv < -step: target = prev - step
            return target
        v = max(-self.v_max, min(self.v_max, v))
        w = max(-self.w_max, min(self.w_max, w))
        v = limit(self.v_prev, v, self.a_v)
        w = limit(self.w_prev, w, self.a_w)
        t = Twist(); t.linear.x = float(v); t.angular.z = float(w)
        self.pub_cmd.publish(t)
        self.v_prev, self.w_prev = v, w

    def _publish_goal_achieved_once(self):
        if not self._goal_announced:
            self.pub_goal_achieved.publish(Bool(data=True))
            self._goal_announced = True
            self.get_logger().info("Published /goal_achieved = true")

    def goal_in_working_now(self) -> Optional[PoseStamped]:
        """Transform the latest goal into the working frame using *current* TF each tick."""
        goal = self.goal_msg
        if goal is None:
            return None
        src = goal.header.frame_id or self.working_frame
        dst = self.working_frame
        if src == dst:
            return goal
        try:
            tf = self.tf_buffer.lookup_transform(dst, src, Time(), timeout=Duration(seconds=0.05))
            ps = PoseStamped()
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.header.frame_id = dst
            ps.pose = do_transform_pose(goal.pose, tf)
            return ps
        except (LookupException, ConnectivityException, ExtrapolationException):
            if self.assume_equal and src == 'map' and dst == 'odom':
                ps = PoseStamped()
                ps.header.stamp = self.get_clock().now().to_msg()
                ps.header.frame_id = dst
                ps.pose = goal.pose
                return ps
            return None


# ------------------------ Behaviours ------------------------

class WaitForPose(py_trees.behaviour.Behaviour):
    def __init__(self, n: GoToGoalNode): super().__init__('WaitForPose'); self.n=n
    def update(self): return py_trees.common.Status.SUCCESS if self.n.current_pose else py_trees.common.Status.RUNNING

class WaitForGoal(py_trees.behaviour.Behaviour):
    def __init__(self, n: GoToGoalNode): super().__init__('WaitForGoal'); self.n=n
    def update(self):
        if self.n.goal_msg is None or self.n.goal_reached:
            self.n.stop(); return py_trees.common.Status.RUNNING
        return py_trees.common.Status.SUCCESS

class TransformGoal(py_trees.behaviour.Behaviour):
    """Ensure TF is available once; the controllers still re-transform every tick."""
    def __init__(self, n: GoToGoalNode): super().__init__('EnsureTF'); self.n=n
    def update(self):
        ok = self.n.goal_in_working_now() is not None
        if ok: return py_trees.common.Status.SUCCESS
        self.n.stop(); return py_trees.common.Status.RUNNING

class DriveToGoal(py_trees.behaviour.Behaviour):
    def __init__(self, n: GoToGoalNode): super().__init__('DriveToGoal'); self.n=n
    def update(self):
        pose = self.n.current_pose
        goal = self.n.goal_in_working_now()  # re-transform EVERY TICK
        if pose is None or goal is None:
            self.n.stop(); return py_trees.common.Status.FAILURE
        x,y,yaw = pose; gx,gy = goal.pose.position.x, goal.pose.position.y
        dx,dy = gx-x, gy-y; rho = math.hypot(dx,dy)
        if rho <= self.n.pos_tol:
            self.n.publish_cmd(0.0,0.0); return py_trees.common.Status.SUCCESS
        theta = math.atan2(dy,dx); alpha = angle_wrap(theta - yaw)
        v = self.n.k_rho * rho; w = self.n.k_alpha * alpha
        self.n.publish_cmd(v,w); return py_trees.common.Status.RUNNING

class AlignFinalYaw(py_trees.behaviour.Behaviour):
    def __init__(self, n: GoToGoalNode): super().__init__('AlignFinalYaw'); self.n=n
    def update(self):
        pose = self.n.current_pose
        goal = self.n.goal_in_working_now()  # re-transform EVERY TICK
        if pose is None or goal is None:
            self.n.stop(); return py_trees.common.Status.FAILURE
        yaw = pose[2]; yaw_g = yaw_from_quaternion(goal.pose.orientation)
        e = angle_wrap(yaw_g - yaw)
        if abs(e) <= self.n.yaw_tol:
            self.n.publish_cmd(0.0,0.0); self.n.goal_reached=True; self.n._publish_goal_achieved_once()
            return py_trees.common.Status.SUCCESS
        self.n.publish_cmd(0.0, self.n.k_yaw * e)
        return py_trees.common.Status.RUNNING

class Stop(py_trees.behaviour.Behaviour):
    def __init__(self, n: GoToGoalNode): super().__init__('Stop'); self.n=n
    def update(self): self.n.stop(); return py_trees.common.Status.SUCCESS


def make_tree(n: GoToGoalNode) -> py_trees.behaviour.Behaviour:
    root = py_trees.composites.Sequence(name='Navigate', memory=True)
    approach = py_trees.composites.Sequence(name='ApproachThenAlign', memory=True)
    wait_pose = WaitForPose(n); wait_goal = WaitForGoal(n); ensure_tf = TransformGoal(n)
    drive = DriveToGoal(n); align = AlignFinalYaw(n); stop = Stop(n)
    approach.add_children([drive, align])
    root.add_children([wait_pose, wait_goal, ensure_tf, approach, stop])
    return root


def main():
    rclpy.init()
    node = GoToGoalNode()
    root = make_tree(node)
    bt = py_trees_ros.trees.BehaviourTree(root)
    bt.setup(node=node, node_name='bt_tree')
    bt.tick_tock(period_ms=50)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop(); node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
