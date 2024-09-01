import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import numpy as np
import math
import heapq
import time

class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.g = float('inf')
        self.rhs = float('inf')
        self.key = [float('inf'), float('inf')]

    def __lt__(self, other):
        return self.key < other.key

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

class DStar:
    def __init__(self, start, goal, grid, resolution):
        self.start = State(start[0], start[1])
        self.goal = State(goal[0], goal[1])
        self.grid = grid
        self.resolution = resolution
        self.U = []
        self.km = 0
        self.states = {}

    def heuristic(self, s1, s2):
        return abs(s1.x - s2.x) + abs(s1.y - s2.y)

    def calculate_key(self, s):
        return [min(s.g, s.rhs) + self.heuristic(s, self.start) + self.km, min(s.g, s.rhs)]

    def get_neighbors(self, s):
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            x, y = s.x + dx, s.y + dy
            if 0 <= x < self.grid.shape[1] and 0 <= y < self.grid.shape[0] and self.grid[y, x] == 0:
                neighbors.append(State(x, y))
        return neighbors

    def get_state(self, x, y):
        if (x, y) not in self.states:
            self.states[(x, y)] = State(x, y)
        return self.states[(x, y)]

    def update_vertex(self, u):
        if u != self.goal:
            u.rhs = min([self.get_state(s.x, s.y).g + self.resolution for s in self.get_neighbors(u)])
        if u in self.U:
            self.U.remove(u)
        if u.g != u.rhs:
            u.key = self.calculate_key(u)
            heapq.heappush(self.U, u)

    def compute_shortest_path(self):
        while self.U and (self.U[0].key < self.calculate_key(self.start) or self.start.rhs > self.start.g):
            u = heapq.heappop(self.U)
            if u.key < self.calculate_key(u):
                u.key = self.calculate_key(u)
                heapq.heappush(self.U, u)
            elif u.g > u.rhs:
                u.g = u.rhs
                for s in self.get_neighbors(u):
                    self.update_vertex(self.get_state(s.x, s.y))
            else:
                u.g = float('inf')
                self.update_vertex(u)
                for s in self.get_neighbors(u):
                    self.update_vertex(self.get_state(s.x, s.y))

    def plan(self):
        self.start.rhs = 0
        self.start.key = self.calculate_key(self.start)
        heapq.heappush(self.U, self.start)
        self.compute_shortest_path()
        
        path = []
        current = self.start
        while current != self.goal:
            path.append([current.x, current.y])
            neighbors = self.get_neighbors(current)
            current = min(neighbors, key=lambda s: self.get_state(s.x, s.y).g + self.resolution)
        path.append([self.goal.x, self.goal.y])
        return path

    def update_grid(self, new_grid):
        changed_states = []
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                if self.grid[y, x] != new_grid[y, x]:
                    changed_states.append(self.get_state(x, y))
        self.grid = new_grid
        self.km += self.heuristic(self.start, self.goal)
        for s in changed_states:
            self.update_vertex(s)
        self.compute_shortest_path()

class DStarExploration(Node):
    def __init__(self):
        super().__init__('DStar_Exploration')
        
        # Load parameters
        self.declare_parameter('lookahead_distance', 1.0)
        self.declare_parameter('speed', 0.2)
        self.declare_parameter('robot_radius', 0.2)

        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.speed = self.get_parameter('speed').value
        self.robot_radius = self.get_parameter('robot_radius').value

        # Initialize subscribers
        self.create_subscription(OccupancyGrid, 'map', self.map_callback, 10)
        self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        
        # Initialize publishers
        self.vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_publisher = self.create_publisher(String, 'exploration_status', 10)

        # Initialize D* parameters
        self.dstar = None
        self.path = None
        self.current_goal = None
        self.exploration_start_time = None
        self.exploration_end_time = None
        self.exploration_completed = False
        self.explored_area = 0
        self.total_area = 0
        self.grid = None

        # Create a timer for checking exploration status
        self.create_timer(1.0, self.check_exploration_status)

        self.get_logger().info("D* Exploration Node Initialized")

    def map_callback(self, msg):
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        
        self.grid = np.array(msg.data).reshape((self.height, self.width))
        self.total_area = self.width * self.height * self.resolution**2

        if self.dstar is None and hasattr(self, 'current_pose'):
            self.exploration_start_time = time.time()
            start = [int((self.current_pose.x - self.origin_x) / self.resolution),
                     int((self.current_pose.y - self.origin_y) / self.resolution)]
            goal = self.select_exploration_goal()
            self.dstar = DStar(start, goal, self.grid, self.resolution)
            self.path = self.dstar.plan()
        else:
            self.dstar.update_grid(self.grid)
            self.path = self.dstar.plan()

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose.position
        if self.path:
            self.pure_pursuit(self.path)

    def scan_callback(self, msg):
        if min(msg.ranges) < self.robot_radius:
            self.avoid_obstacle(msg)

    def select_exploration_goal(self):
        frontiers = self.detect_frontiers()
        if frontiers:
            return max(frontiers, key=lambda f: f[2])[:2]  # Return the frontier with highest information gain
        else:
            return self.generate_random_goal()

    def detect_frontiers(self):
        frontiers = []
        robot_size_cells = int(self.robot_radius / self.resolution)
        for i in range(robot_size_cells, self.height - robot_size_cells):
            for j in range(robot_size_cells, self.width - robot_size_cells):
                if self.is_frontier(i, j) and self.is_robot_sized_space_free(i, j, robot_size_cells):
                    x = j * self.resolution + self.origin_x
                    y = i * self.resolution + self.origin_y
                    info_gain = self.calculate_information_gain(i, j)
                    frontiers.append((x, y, info_gain))
        return frontiers

    def is_frontier(self, i, j):
        if self.grid[i][j] == -1:
            neighbors = [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
            return any(0 <= ni < self.height and 0 <= nj < self.width and self.grid[ni][nj] == 0 for ni, nj in neighbors)
        return False

    def is_robot_sized_space_free(self, i, j, robot_size_cells):
        for di in range(-robot_size_cells, robot_size_cells + 1):
            for dj in range(-robot_size_cells, robot_size_cells + 1):
                if not (0 <= i + di < self.height and 0 <= j + dj < self.width):
                    return False
                if self.grid[i + di][j + dj] > 50:  # obstacle
                    return False
        return True

    def calculate_information_gain(self, i, j):
        info_gain = 0
        for di in range(-5, 6):
            for dj in range(-5, 6):
                ni, nj = i + di, j + dj
                if 0 <= ni < self.height and 0 <= nj < self.width and self.grid[ni][nj] == -1:
                    info_gain += 1
        return info_gain

    def generate_random_goal(self):
        x = int(np.random.uniform(0, self.width))
        y = int(np.random.uniform(0, self.height))
        return [x, y]

    def pure_pursuit(self, path):
        if not path:
            return

        lookahead_point = None
        for point in reversed(path):
            x = point[0] * self.resolution + self.origin_x
            y = point[1] * self.resolution + self.origin_y
            distance = math.hypot(x - self.current_pose.x, y - self.current_pose.y)
            if distance <= self.lookahead_distance:
                lookahead_point = [x, y]
                break

        if lookahead_point is None:
            return

        alpha = math.atan2(lookahead_point[1] - self.current_pose.y, lookahead_point[0] - self.current_pose.x)
        curvature = 2.0 * math.sin(alpha) / self.lookahead_distance

        twist = Twist()
        twist.linear.x = self.speed
        twist.angular.z = curvature * self.speed

        self.vel_publisher.publish(twist)

    def avoid_obstacle(self, scan_msg):
        twist = Twist()
        if min(scan_msg.ranges) < self.robot_radius:
            twist.linear.x = 0.0
            twist.angular.z = 0.5  # Turn in place
        else:
            twist.linear.x = self.speed
            twist.angular.z = 0.0
        self.vel_publisher.publish(twist)

    def check_exploration_status(self):
        if not self.exploration_completed and self.check_exploration_complete():
            self.exploration_end_time = time.time()
            self.exploration_completed = True
            self.display_exploration_time()

    def check_exploration_complete(self):
        time_limit = 300  # 5 minutes
        area_threshold = 0.9  # 90% of the area

        if self.exploration_start_time and time.time() - self.exploration_start_time > time_limit:
            return True
        if self.total_area > 0 and self.explored_area / self.total_area > area_threshold:
            return True
        return False

    def display_exploration_time(self):
        if self.exploration_start_time and self.exploration_end_time:
            total_time = self.exploration_end_time - self.exploration_start_time
            message = f"Exploration completed in {total_time:.2f} seconds\n"
            if self.total_area > 0:
                message += f"Explored area: {self.explored_area:.2f} m² ({self.explored_area/self.total_area*100:.2f}% of total area)"
            else:
                message += f"Explored area: {self.explored_area:.2f} m²"
            
            # Log the message
            self.get_logger().info(message)
            
            # Publish the message to a topic
            status_msg = String()
            status_msg.data = message
            self.status_publisher.publish(status_msg)
        else:
            self.get_logger().warning("Exploration time cannot be calculated")

def main(args=None):
    rclpy.init(args=args)
    dstar_exploration = DStarExploration()
    rclpy.spin(dstar_exploration)
    dstar_exploration.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
