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
from rclpy.parameter import Parameter

import tf2_ros
from tf2_ros import TransformException

class MapDisplayNode(Node):
    def __init__(self):
        super().__init__('map_display_node')
        
        self.set_parameters([Parameter('use_sim_time', value=True)])

        self.pole_przeszkody = 8
        self.min_klaster_rozmiar_pocz = 5
        self.min_klaster_rozmiar = self.min_klaster_rozmiar_pocz
        self.min_goal_distance = 30
        self.frontier_min_unknown_neighbors = 3

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
        self.goal_publisher = self.create_publisher(PoseStamped, '/destination_goal', goal_qos_profile)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/rosbot_base_controller/odom', self.odom_callback, 10)
        self.map_fields_pub = self.create_publisher(OccupancyGrid, '/map_fields', 10)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.stuck_timer = self.create_timer(self.stuck_check_interval, self.check_if_stuck_callback)
        self.get_logger().info('Węzeł MapDisplayNode uruchomiony z poprawioną eksploracją frontier.')

    def odom_callback(self, msg):
        if self.mission_completed: return
        from_frame = 'map'
        to_frame = 'base_link'

        try:
            t = self.tf_buffer.lookup_transform(from_frame, to_frame, rclpy.time.Time())
            
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
        
        if self.map_info:
            self.update_plot()

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
                is_current_goal_removed = self.exploration_points and self.exploration_points[0] in points_to_remove
                self.exploration_points = [p for p in self.exploration_points if p not in points_to_remove]
                if is_current_goal_removed:
                    self.publish_next_goal()

        self.update_plot()

    def replan_and_sort_goals(self):
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
                for (cx, cy), gain in targets_with_info:
                    distance = np.sqrt((cx - robot_grid_x)**2 + (cy - robot_grid_y)**2)
                    scored_targets.append(((cx, cy), distance, gain))
                
                scored_targets.sort(key=lambda item: (item[1], -item[2]))
                
                filtered_goals = []
                for (cx, cy), distance, gain in scored_targets:
                    is_too_close = False
                    for (fx, fy) in filtered_goals:
                        if np.sqrt((cx - fx)**2 + (cy - fy)**2) < self.min_goal_distance:
                            is_too_close = True
                            break
                    if not is_too_close:
                        if self._check_path_validity_bfs(robot_grid_x, robot_grid_y, cx, cy, self.map_data_raw):
                            filtered_goals.append((cx, cy))
                        else:
                            self.get_logger().debug(f"Goal ({cx}, {cy}) jest nieosiągalny z pozycji robota. Pomijam.")
                
                self.exploration_points = filtered_goals
                self.get_logger().info(f"Top 5 celów po sortowaniu: {[f'({g[0][0]}, {g[0][1]}) dist:{g[1]:.2f} gain:{g[2]}' for g in scored_targets[:5]]}")
        
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

        fields_msg = OccupancyGrid()
        fields_msg.header.stamp = self.get_clock().now().to_msg()
        fields_msg.header.frame_id = "map"
        fields_msg.info = self.map_info
        fields_msg.data = self.map_fields_raw.flatten().tolist()
        self.map_fields_pub.publish(fields_msg)

    def plan_frontier_points_via_clustering(self, map_data):
        """
        Find frontier points: free cells (0) adjacent to unknown cells (-1)
        that are reachable without crossing obstacles.
        """
        height, width = map_data.shape
        all_frontier_points = []

        for r in range(1, height - 1):
            for c in range(1, width - 1):
                if map_data[r, c] != 0:
                    continue
                
                unknown_neighbors = 0
                has_obstacle_neighbor = False
                
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < height and 0 <= nc < width:
                            if map_data[nr, nc] == -1:
                                unknown_neighbors += 1
                            elif map_data[nr, nc] >= 99:  
                                has_obstacle_neighbor = True
                
                if unknown_neighbors >= self.frontier_min_unknown_neighbors and not has_obstacle_neighbor:
                    all_frontier_points.append(((c, r), unknown_neighbors))
        
        self.get_logger().info(f"Znaleziono {len(all_frontier_points)} punktów frontier.")
        return all_frontier_points

    def _check_path_validity_bfs(self, start_x, start_y, end_x, end_y, map_data):
        """
        Use BFS to check if there's a valid path through free space (0)
        from start to end position. More robust than line-of-sight.
        """
        height, width = map_data.shape
        
        if not (0 <= start_y < height and 0 <= start_x < width):
            return False
        if not (0 <= end_y < height and 0 <= end_x < width):
            return False
        
        if map_data[start_y, start_x] != 0:
            return False
        if map_data[end_y, end_x] != 0:
            return False
        
        visited = np.zeros((height, width), dtype=bool)
        queue = deque([(start_x, start_y)])
        visited[start_y, start_x] = True
        
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        max_search_distance = int(np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2) * 1.5)
        search_count = 0
        max_iterations = 10000
        
        while queue and search_count < max_iterations:
            x, y = queue.popleft()
            search_count += 1
            
            if x == end_x and y == end_y:
                return True
            
            if abs(x - start_x) > max_search_distance or abs(y - start_y) > max_search_distance:
                continue
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < width and 0 <= ny < height:
                    if not visited[ny, nx] and map_data[ny, nx] == 0:
                        visited[ny, nx] = True
                        queue.append((nx, ny))
        
        return False

    def publish_next_goal(self):
        self.time_without_movement = 0.0
        self.last_known_position = None
        
        if not self.exploration_points:
            if self.exploration_phase == 1:
                self.get_logger().warn("Faza 1 zakończona. Brak dużych celów. Przechodzę do Fazy 2 (mniejsze cele).")
                self.exploration_phase = 2
                self.min_klaster_rozmiar = int(self.min_klaster_rozmiar_pocz / 2)
                self.frontier_min_unknown_neighbors = 1  
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
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = "map"
        goal_msg.pose.position.x = target_world_x
        goal_msg.pose.position.y = target_world_y
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.w = 1.0
        self.goal_publisher.publish(goal_msg)
        self.get_logger().info(f"Opublikowano nowy cel frontier: ({target_world_x:.2f}, {target_world_y:.2f})")

    def is_area_fully_explored(self, center_x, center_y, map_data, radius=10):
        min_r = max(0, center_y - radius)
        max_r = min(map_data.shape[0], center_y + radius + 1)
        min_c = max(0, center_x - radius)
        max_c = min(map_data.shape[1], center_x + radius + 1)
        area = map_data[min_r:max_r, min_c:max_c]
        return not np.any(area == -1)

    def update_plot(self):
        if self.map_data_raw is None or self.map_fields_raw is None or self.map_info is None:
            return
        
        self.ax1.clear()
        self.ax2.clear()
        
        display_map_data = np.vectorize(self.display_mapping_map.get)(self.map_data_raw)
        self.ax1.imshow(display_map_data, cmap=self.cmap_map, vmin=0, vmax=len(self.display_mapping_map)-1, origin='lower')
        self.ax1.set_title(f'/map ({self.map_info.width}x{self.map_info.height})')
        self.ax1.set_xlabel('X (komórki)')
        self.ax1.set_ylabel('Y (komórki)')
        
        display_costmap_data = np.vectorize(self.display_mapping_costmap.get)(self.map_fields_raw)
        self.ax2.imshow(display_costmap_data, cmap=self.cmap_fields, vmin=0, vmax=len(self.display_mapping_costmap)-1, origin='lower')
        
        if self.mission_completed:
            self.ax2.set_title('Misja Zakończona!')
        else:
            self.ax2.set_title(f'Costmap (Faza {self.exploration_phase}, Pozostało {len(self.exploration_points)} celów)')
        
        self.ax2.set_xlabel('X (komórki)')
        self.ax2.set_ylabel('Y (komórki)')

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
        
        self.ax2.legend(loc='upper right')
        plt.draw()
        plt.pause(0.01)

    def convert_grid_to_world(self, grid_x, grid_y):
        world_x = self.map_info.origin.position.x + (grid_x + 0.5) * self.map_info.resolution
        world_y = self.map_info.origin.position.y + (grid_y + 0.5) * self.map_info.resolution
        return world_x, world_y

    def on_motion(self, event):
        if event.inaxes in [self.ax1, self.ax2] and self.map_info:
            col = int(event.xdata)
            row = int(event.ydata)
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