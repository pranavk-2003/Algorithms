class RRTExploration(Node):
    def __init__(self):
        super().__init__('RRT_Exploration')
        
        # Load parameters
        with open("src/autonomous/config/params.yaml", 'r') as file:
            params = yaml.safe_load(file)

        self.lookahead_distance = params["lookahead_distance"]
        self.speed = params["speed"]
        self.expansion_size = params["expansion_size"]
        self.target_error = params["target_error"]
        self.robot_radius = params["robot_r"]

        # Initialize subscribers
        self.create_subscription(OccupancyGrid, 'map', self.map_callback, 10)
        self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        
        # Initialize publishers
        self.vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_publisher = self.create_publisher(String, 'exploration_status', 10)

        # Initialize RRT parameters
        self.rrt = None
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

        self.get_logger().info("RRT Exploration Node Initialized")

    def map_callback(self, msg):
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        
        self.grid = np.array(msg.data).reshape((self.height, self.width))
        self.obstacle_list = self.occupancy_grid_to_obstacle_list(msg)
        self.total_area = self.width * self.height * self.resolution**2

        if self.rrt is None and hasattr(self, 'current_pose') and not self.exploration_start_time:
            self.exploration_start_time = time.time()
            start = [self.current_pose.x, self.current_pose.y]
            goal = self.select_exploration_goal()
            self.rrt = RRT(start, goal, self.obstacle_list, 
                           [self.origin_x, self.origin_x + self.width * self.resolution])
            self.path = self.rrt.planning()

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose.position
        if self.path:
            self.pure_pursuit(self.path)

    def scan_callback(self, msg):
        # Removed obstacle avoidance logic
        pass

    def occupancy_grid_to_obstacle_list(self, grid_msg):
        obstacle_list = []
        self.explored_area = 0
        for i in range(self.height):
            for j in range(self.width):
                if grid_msg.data[i * self.width + j] > 50:  # Assuming > 50 is obstacle
                    x = j * self.resolution + self.origin_x
                    y = i * self.resolution + self.origin_y
                    obstacle_list.append((x, y, self.resolution / 2))
                elif grid_msg.data[i * self.width + j] == 0:  # Assuming 0 is explored free space
                    self.explored_area += self.resolution**2
        return obstacle_list

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
        x = random.uniform(self.origin_x, self.origin_x + self.width * self.resolution)
        y = random.uniform(self.origin_y, self.origin_y + self.height * self.resolution)
        return [x, y]

    def pure_pursuit(self, path):
        if not path:
            return

        lookahead_point = None
        for point in reversed(path):
            distance = math.hypot(point[0] - self.current_pose.x, point[1] - self.current_pose.y)
            if distance <= self.lookahead_distance:
                lookahead_point = point
                break

        if lookahead_point is None:
            lookahead_point = path[-1]

        alpha = math.atan2(lookahead_point[1] - self.current_pose.y, lookahead_point[0] - self.current_pose.x)
        curvature = 2.0 * math.sin(alpha) / self.lookahead_distance

        twist = Twist()
        twist.linear.x = self.speed
        twist.angular.z = curvature * self.speed

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
