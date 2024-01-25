import numpy as np

# from src.environment.RandomEnv import SimEnv
from src.sim.Functions import rotate_vector, line_intersection, is_between, plot_detection, angle_between
from numba import jit

class Robot:
    def __init__(self, position, angle_range=np.pi / 3, num_vectors=5, perception_range=50, max_vel=5, avoid_range=7,
                 arrival_range=5):
        self.last_node = None
        self.position = np.array(position)
        self.velocity = np.random.randint(-360.0, 360.0, size=(2,))
        velocity_magnitude = np.linalg.norm(self.velocity)
        # Initializing a very small start velocity for initial orientation and turning
        self.velocity = (self.velocity / velocity_magnitude) * 0.001
        self.acceleration = np.array([0.0, 0.0])
        self.orientation = np.copy(self.velocity)  # np.array([1, 0])
        self.angle_range = angle_range
        self.num_vectors = num_vectors
        self.perception_range = perception_range
        self.perception_cone = self.get_perception_cone()
        self.max_vel = max_vel
        self.avoid_range = avoid_range
        self.position_history = [np.copy(self.position)]
        self.orientation_history = [np.copy(self.orientation)]
        self.velocity_history = [np.copy(self.velocity)]
        self.good_position_history = []
        self.is_avoiding = False
        self.path = None
        self.path_len = 1
        self.path_history = []
        self.arrival_range = arrival_range

    def update_velocity(self):
        new_velocity = self.velocity + self.acceleration
        velocity_magnitude = np.linalg.norm(new_velocity)

        if velocity_magnitude > self.max_vel:
            new_velocity = (new_velocity / velocity_magnitude) * self.max_vel

        vel_turn_rate = 0.1
        if self.is_avoiding:
            vel_turn_rate = 1
        self.velocity = self.smooth_transition(self.velocity, new_velocity, vel_turn_rate)

        # Only update orientation if velocity is nonzero
        if np.any(self.velocity != 0):
            if np.any(self.velocity != 0):
                new_orientation = self.velocity / np.linalg.norm(self.velocity)
                self.orientation = self.smooth_turn(self.orientation, new_orientation, 0.5)
        self.perception_cone = self.get_perception_cone()

    def smooth_turn(self, current_orientation, new_orientation, turning_rate):

        return current_orientation * (1 - turning_rate) + new_orientation * turning_rate

    def smooth_transition(self, current_value, target_value, turning_rate):

        return current_value * (1 - turning_rate) + target_value * turning_rate

    def update_position(self):
        if self.path is not None and len(self.path) > 1:
            arrival_range = self.arrival_range
        else:
            arrival_range = self.perception_range / 2
        self.position += self.velocity
        self.position_history.append(np.copy(self.position))
        self.orientation_history.append(np.copy(self.orientation))
        self.velocity_history.append(np.copy(self.velocity))
        if self.path is not None and len(self.path) > 0 and\
                np.linalg.norm(self.position - self.path[0]) < arrival_range:
            self.last_node = self.path[-1]
            self.path = self.path[1:]

    # Generate vectors in a cone around the orientation vector.
    def get_perception_cone(self):
        orientation_angle = np.arctan2(self.orientation[1], self.orientation[0])
        start_angle = orientation_angle - self.angle_range / 2
        end_angle = orientation_angle + self.angle_range / 2
        angles = np.linspace(start_angle, end_angle, self.num_vectors)

        vectors = rotate_vector(self.orientation, angles - orientation_angle)

        # Normalize the vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized_vectors = vectors / norms

        # Scale the vectors
        scaled_vectors = normalized_vectors * self.perception_range

        return scaled_vectors

    # Detects closest intersections with polygons in the perception cone.
    def detect(self, polygons):
        closest_intersections = []
        open_space_points = []

        for vector in self.perception_cone:
            closest_point = None
            min_distance = np.inf

            for polygon in polygons:
                for i in range(len(polygon)):
                    line1 = [self.position, self.position + vector]
                    line2 = [polygon[i], polygon[(i + 1) % len(polygon)]]

                    intersection = line_intersection(line1, line2)
                    if intersection and is_between(line2[0], line2[1], intersection):
                        to_intersection = np.array(intersection) - self.position
                        if (np.linalg.norm(to_intersection) < np.linalg.norm(vector) and angle_between(vector,
                                                                                                       to_intersection)
                                <= np.pi / 2):
                            distance = np.linalg.norm(to_intersection)

                            if distance < min_distance:
                                min_distance = distance
                                closest_point = intersection

            if closest_point:
                closest_intersections.append(closest_point)
                x_values = np.linspace(self.position[0], closest_point[0], 101)
                y_values = np.linspace(self.position[1], closest_point[1], 101)
                x_values = x_values[:-1]
                y_values = y_values[:-1]

                points = np.column_stack((x_values, y_values))
                open_space_points.extend(points)
            else:
                x_values = np.linspace(self.position[0], self.position[0] + vector[0], 100)
                y_values = np.linspace(self.position[1], self.position[1] + vector[1], 100)
                points = np.column_stack((x_values, y_values))
                open_space_points.extend(points)

        return closest_intersections, open_space_points

# env = SimEnv(width=250, height=250, min_room_size=25, max_room_size=50, min_rooms=20, max_rooms=20, hallway_width=5,
#              n_robots=5, r_radius=2, rand_connections=0)
# env.print_grid()
# env.draw_env('env.png')
# # obstacles = env.get_obstacles()
# env.scale_grid(1000, 1000)
# polygons = env.convert_to_poly()
# test_rob = Robot(env.starting_points[0])
# test_detect = test_rob.detect(polygons)
# plot_detection(test_rob.position, test_rob.perception_cone, polygons, test_detect)

# print('test')
