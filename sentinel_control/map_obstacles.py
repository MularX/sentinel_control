#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_dilation, binary_fill_holes
from collections import deque
import time

import tf2_ros
from tf2_ros import TransformException

class MapDisplayNode(Node):
    def __init__(self):
        super().__init__('map_display_node')


        self.pole_przeszkody = 10
        self.min_klaster_rozmiar_pocz = 20 # Poczatkowy, wysoki prog
        self.min_klaster_rozmiar = self.min_klaster_rozmiar_pocz
        self.kara_za_odleglosc = 2.5 


        self.exploration_phase = 1 
        self.mission_completed = False
        self.initial_robot_position = None 

      
        self.stuck_check_interval = 1.0
        self.stuck_timeout = 5.0
        self.last_known_position = None
        self.time_without_movement = 0.0

        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 7))
        self.fig.suptitle('Wizualizacje Map')
        self.ax1.set_title('/map')
        self.ax2.set_title('/map_fields (Costmap)')

        self.map_info = None
        self.robot_x, self.robot_y = None, None
        self.map_data_raw, self.map_fields_raw = None, None
        self.exploration_points = []

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.cmap_fields = ListedColormap(['gray', 'green', 'orange', 'red'])
        self.display_mapping_costmap = {-1: 0, 0: 1, 99: 2, 100: 3}
        self.cmap_map = ListedColormap(['gray', 'green', 'red'])
        self.display_mapping_map = {-1: 0, 0: 1, 100: 2}

        goal_qos_profile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self.goal_publisher = self.create_publisher(PoseStamped, '/exploration_goal', goal_qos_profile)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/rosbot_base_controller/odom', self.odom_callback, 10)
        self.map_fields_pub = self.create_publisher(OccupancyGrid, '/map_fields', 10)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.stuck_timer = self.create_timer(self.stuck_check_interval, self.check_if_stuck_callback)
        self.get_logger().info('Węzeł MapDisplayNode uruchomiony z zaawansowanymi funkcjami.')

    def check_if_stuck_callback(self):
        if self.robot_x is None or not self.exploration_points or self.mission_completed:
            return

        current_position = (self.robot_x, self.robot_y)
        if self.last_known_position is None:
            self.last_known_position = current_position
            return

        distance_moved = np.sqrt((current_position[0] - self.last_known_position[0])**2 + (current_position[1] - self.last_known_position[1])**2)

        if distance_moved > 0.1:
            self.time_without_movement = 0.0
            self.last_known_position = current_position
        else:
            self.time_without_movement += self.stuck_check_interval

        if self.time_without_movement >= self.stuck_timeout:
            self.get_logger().warn(f"Robot nie poruszył się od {self.stuck_timeout}s. Uznaję, że utknął!")
            self.get_logger().warn(f"Odrzucam aktualny cel: {self.exploration_points[0]}")
            
            self.exploration_points.pop(0)
            self.publish_next_goal()
            self.time_without_movement = 0.0
            self.last_known_position = None

    def odom_callback(self, msg):
        if self.mission_completed: return
        from_frame = 'map'
        to_frame = 'base_link'

        try:
            
            t = self.tf_buffer.lookup_transform(from_frame, to_frame, msg.header.stamp)
            self.robot_x = t.transform.translation.x
            self.robot_y = t.transform.translation.y

            
            if self.initial_robot_position is None:
                self.initial_robot_position = (self.robot_x, self.robot_y)
                self.get_logger().info(f"Zapisano pozycję startową robota: {self.initial_robot_position}")
         

            if self.exploration_points and self.map_info:
                goal_x, goal_y = self.exploration_points[0]
                world_x, world_y = self.convert_grid_to_world(goal_x, goal_y)
                distance_to_goal = np.sqrt((self.robot_x - world_x)**2 + (self.robot_y - world_y)**2)
                if distance_to_goal < 0.5:
                    self.get_logger().info(f"Cel {self.exploration_points[0]} osiągnięty! Publikowanie następnego.")
                    self.exploration_points.pop(0)
                    self.publish_next_goal()
                    self.time_without_movement = 0.0
                    self.last_known_position = None

        except TransformException as ex:
            self.get_logger().warn(f'Nie udało się uzyskać transformacji: {ex}', throttle_duration_sec=1.0)
            return

        if self.map_info: self.update_plot()

    def map_callback(self, msg):
        if self.mission_completed: return
        map_structure_changed = False
        if self.map_info is None or \
           msg.info.width != self.map_info.width or msg.info.height != self.map_info.height or \
           msg.info.origin.position.x != self.map_info.origin.position.x or msg.info.origin.position.y != self.map_info.origin.position.y:
            map_structure_changed = True
            self.get_logger().warn("!!! WYKRYTO ZMIANĘ STRUKTURY MAPY !!!")

        self.map_info = msg.info
        self.map_data_raw = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.update_and_publish_costmap()

        if map_structure_changed:
            self.get_logger().warn("Resetowanie planu eksploracji (Faza 1).")
            self.exploration_phase = 1
            self.min_klaster_rozmiar = self.min_klaster_rozmiar_pocz
            self.replan_and_sort_goals()
        
        elif self.exploration_points:
            points_to_remove = []
            for point in self.exploration_points:
                if self.is_area_fully_explored(point[0], point[1], self.map_data_raw, radius=15):
                    points_to_remove.append(point)
            if points_to_remove:
                self.get_logger().info(f"Usunięto {len(points_to_remove)} odkrytych celów.")
                is_current_goal_removed = self.exploration_points[0] in points_to_remove
                self.exploration_points = [p for p in self.exploration_points if p not in points_to_remove]
                if is_current_goal_removed:
                    self.publish_next_goal()

        self.update_plot()

    def replan_and_sort_goals(self):
        """Nowa funkcja do centralnego zarządzania planowaniem."""
        targets_with_info = self.plan_frontier_points_via_clustering(self.map_fields_raw)

        if not targets_with_info:
            self.exploration_points = []
        else:
            if self.robot_x is None or self.robot_y is None:
                self.get_logger().warn("Brak pozycji robota, sortowanie tylko wg zysku.")
                targets_with_info.sort(key=lambda item: item[1], reverse=True)
                self.exploration_points = [item[0] for item in targets_with_info]
            else:
                robot_grid_x = int((self.robot_x - self.map_info.origin.position.x) / self.map_info.resolution)
                robot_grid_y = int((self.robot_y - self.map_info.origin.position.y) / self.map_info.resolution)
                
                scored_targets = []
                for (cx, cy), size in targets_with_info:
                    cost = ((cx - robot_grid_x)**2 + (cy - robot_grid_y)**2) / 100.0
                    gain = size
                    utility = gain - self.kara_za_odleglosc * cost
                    scored_targets.append(((cx, cy), utility))
                
                scored_targets.sort(key=lambda item: item[1], reverse=True)
                self.exploration_points = [item[0] for item in scored_targets]

        self.get_logger().info(f'Faza {self.exploration_phase}: Nowy plan, {len(self.exploration_points)} celów.')
        self.publish_next_goal()

    def update_and_publish_costmap(self):
        if self.map_data_raw is None: return
        

        obstacle_mask = (self.map_data_raw == 100)

        filled_obstacle_mask = binary_fill_holes(obstacle_mask)

        struct_element = np.ones((2 * self.pole_przeszkody + 1, 2 * self.pole_przeszkody + 1))
        inflated_mask = binary_dilation(filled_obstacle_mask, structure=struct_element)
        
        costmap = self.map_data_raw.copy()
        costmap[inflated_mask] = 99
        costmap[filled_obstacle_mask] = 100
        self.map_fields_raw = costmap

        fields_msg = OccupancyGrid(); fields_msg.header.stamp = self.get_clock().now().to_msg()
        fields_msg.header.frame_id = "map"; fields_msg.info = self.map_info
        fields_msg.data = self.map_fields_raw.flatten().tolist()
        self.map_fields_pub.publish(fields_msg)

    def plan_frontier_points_via_clustering(self, map_data):
        height, width = map_data.shape; frontier_points = []
        for r in range(1, height - 1):
            for c in range(1, width - 1):
                if map_data[r, c] == 0 and np.any(map_data[r-1:r+2, c-1:c+2] == -1):
                    frontier_points.append((c, r))
        if not frontier_points: return []
        
        clusters = []; visited = set(frontier_points); q = deque()

        points_to_visit = set(frontier_points)
        
        for point in frontier_points:
            if point in points_to_visit:
                new_cluster = []; q.append(point); points_to_visit.remove(point)
                while q:
                    current_p = q.popleft(); new_cluster.append(current_p)
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            neighbor = (current_p[0] + dx, current_p[1] + dy)
                            if neighbor in points_to_visit:
                                points_to_visit.remove(neighbor); q.append(neighbor)
                clusters.append(new_cluster)
        
        targets_with_utility_info = []
        for cluster in clusters:
            if len(cluster) >= self.min_klaster_rozmiar:
                sum_x = sum(p[0] for p in cluster); sum_y = sum(p[1] for p in cluster)
                centroid_x = int(sum_x / len(cluster)); centroid_y = int(sum_y / len(cluster))
                
                if map_data[centroid_y, centroid_x] == 0:
                    targets_with_utility_info.append(((centroid_x, centroid_y), len(cluster)))
                else:

                    safe_point = min(cluster, key=lambda p: (p[0]-centroid_x)**2 + (p[1]-centroid_y)**2)
                    targets_with_utility_info.append((safe_point, len(cluster)))
        
        return targets_with_utility_info

    def publish_next_goal(self):
        self.time_without_movement = 0.0
        self.last_known_position = None
        
        if not self.exploration_points:

            if self.exploration_phase == 1:
                self.get_logger().warn("Faza 1 zakończona. Brak dużych celów. Przechodzę do Fazy 2 (mniejsze cele).")
                self.exploration_phase = 2
                self.min_klaster_rozmiar = int(self.min_klaster_rozmiar_pocz / 2) # 
                self.replan_and_sort_goals()
                return
            else: 
                self.get_logger().info("Faza 2 zakończona. Brak dalszych celów do eksploracji.")
                if self.initial_robot_position:
                    self.get_logger().info(f"Misja eksploracji zakończona! Wysyłam robota do punktu startowego: {self.initial_robot_position}")
                    goal_msg = PoseStamped()
                    goal_msg.header.stamp = self.get_clock().now().to_msg()
                    goal_msg.header.frame_id = "map"
                    goal_msg.pose.position.x = self.initial_robot_position[0]
                    goal_msg.pose.position.y = self.initial_robot_position[1]
                    goal_msg.pose.orientation.w = 1.0
                    self.goal_publisher.publish(goal_msg)
                    self.mission_completed = True
                else:
                    self.get_logger().warn("Misja zakończona, ale nie znam pozycji startowej do powrotu.")
                return

        target_grid_x, target_grid_y = self.exploration_points[0]
        target_world_x, target_world_y = self.convert_grid_to_world(target_grid_x, target_grid_y)
        goal_msg = PoseStamped(); goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = "map"; goal_msg.pose.position.x = target_world_x
        goal_msg.pose.position.y = target_world_y; goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.w = 1.0
        self.goal_publisher.publish(goal_msg)
        self.get_logger().info(f"Opublikowano nowy, optymalny cel: ({target_world_x:.2f}, {target_world_y:.2f})")

    # ... reszta kodu bez zmian ...
    def is_area_fully_explored(self, center_x, center_y, map_data, radius=10):
        min_r = max(0, center_y - radius); max_r = min(map_data.shape[0], center_y + radius + 1)
        min_c = max(0, center_x - radius); max_c = min(map_data.shape[1], center_x + radius + 1)
        area = map_data[min_r:max_r, min_c:max_c]
        return not np.any(area == -1)

    def update_plot(self):
        if self.map_data_raw is None or self.map_fields_raw is None or self.map_info is None: return
        self.ax1.clear(); self.ax2.clear()
        display_map_data = np.vectorize(self.display_mapping_map.get)(self.map_data_raw)
        self.ax1.imshow(display_map_data, cmap=self.cmap_map, vmin=0, vmax=len(self.display_mapping_map)-1, origin='lower')
        self.ax1.set_title(f'/map ({self.map_info.width}x{self.map_info.height})')
        self.ax1.set_xlabel('X (komórki)'); self.ax1.set_ylabel('Y (komórki)')
        display_costmap_data = np.vectorize(self.display_mapping_costmap.get)(self.map_fields_raw)
        self.ax2.imshow(display_costmap_data, cmap=self.cmap_fields, vmin=0, vmax=len(self.display_mapping_costmap)-1, origin='lower')
        if self.mission_completed:
            self.ax2.set_title('Misja Zakończona!')
        else:
            self.ax2.set_title(f'Costmap (Faza {self.exploration_phase}, Pozostało {len(self.exploration_points)} celów)')
        self.ax2.set_xlabel('X (komórki)'); self.ax2.set_ylabel('Y (komórki)')
        if self.exploration_points:
            points_x, points_y = zip(*self.exploration_points)
            self.ax2.plot(points_x, points_y, 'b*', markersize=8, label='Pozostałe cele')
            self.ax2.plot(self.exploration_points[0][0], self.exploration_points[0][1], 'r*', markersize=14, label='Aktualny cel')
        if self.robot_x is not None and self.robot_y is not None:
            grid_x = (self.robot_x - self.map_info.origin.position.x) / self.map_info.resolution
            grid_y = (self.robot_y - self.map_info.origin.position.y) / self.map_info.resolution
            if 0 <= grid_x < self.map_info.width and 0 <= grid_y < self.map_info.height:
                self.ax1.plot(grid_x, grid_y, 'ro', markersize=8)
                self.ax2.plot(grid_x, grid_y, 'ro', markersize=8, label='Robot')
        self.ax2.legend(loc='upper right'); plt.draw(); plt.pause(0.01)

    def convert_grid_to_world(self, grid_x, grid_y):
        world_x = self.map_info.origin.position.x + (grid_x + 0.5) * self.map_info.resolution
        world_y = self.map_info.origin.position.y + (grid_y + 0.5) * self.map_info.resolution
        return world_x, world_y

    def on_motion(self, event):
        if event.inaxes in [self.ax1, self.ax2] and self.map_info:
            col = int(event.xdata); row = int(event.ydata)
            if 0 <= col < self.map_info.width and 0 <= row < self.map_info.height:
                map_x, map_y = self.convert_grid_to_world(col, row)
                self.fig.suptitle(f'Kursora: [wiersz: {row}, kol: {col}] | Mapa (x: {map_x:.2f}m, y: {map_y:.2f}m)')
                self.fig.canvas.draw_idle()

def main(args=None):
    rclpy.init(args=args)
    node = MapDisplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.close()

if __name__ == '__main__':
    main()
