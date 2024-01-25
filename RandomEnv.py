import random

import cv2
import numpy as np
import matplotlib.pyplot as plt

import cairo

from QuadTree import QuadTree


class Room:
    """
    Creates a room in the environment given a central position and dimensions
    """

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        # Track if this room is connected
        self.connected = False

    # Determine if this room overlaps with another room
    def intersects(self, other):
        return (self.x < other.x + other.width and self.x + self.width > other.x and
                self.y < other.y + other.height and self.y + self.height > other.y)

    def center(self):
        return self.x + self.width // 2, self.y + self.height // 2


class SimEnv:
    """
    Defines procedurally generated simulation environment, given global dimensions and parameters for random generation
    """

    def __init__(self, width, height, min_room_size, max_room_size, min_rooms, max_rooms, hallway_width, n_robots,
                 r_radius, rand_connections):
        self.quadtree = None
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        self.rooms = []
        self.min_room_size = min_room_size
        self.max_room_size = max_room_size
        self.min_rooms = min_rooms
        self.max_rooms = max_rooms
        self.hallway_width = hallway_width
        self.r_radius = 5
        self.create_rooms()
        self.connect_rooms()
        self.connect_randomly(rand_connections)
        self.rand_connections = rand_connections
        self.starting_points = self.pick_starting_points(n_robots, r_radius)
        self.polygon, self.polygon_arr = self.convert_to_poly()

    def populate_quadtree(self):
        line_id = 0
        for line in self.polygon_arr:
            self.quadtree.insert_line(line, line_id)
            line_id += 1

    def pick_starting_points(self, n_robot, r_radius):
        start_room = random.choice(self.rooms)
        center_x, center_y = start_room.center()
        starting_points = []

        # for _ in range(n_robot):
        #     x = random.randint(center_x - start_room.width // 2 + r_radius, center_x + start_room.width // 2 - r_radius)
        #     y = random.randint(center_y - start_room.height // 2 + r_radius, center_y + start_room.height // 2 -
        #                        r_radius)
        #     starting_points.append((x, y))
        for _ in range(n_robot):
            starting_points.append((center_x, center_y))

        return starting_points

    def create_rooms(self):
        # Set number of rooms
        num_rooms = random.randint(self.min_rooms, self.max_rooms)

        # Set parameters of room
        for _ in range(num_rooms):
            width = random.randint(self.min_room_size, self.max_room_size)
            height = random.randint(self.min_room_size, self.max_room_size)
            x = random.randint(0, self.width - width - 1)
            y = random.randint(0, self.height - height - 1)

            new_room = Room(x, y, width, height)

            # Make sure room does not intersect
            if not any(r.intersects(new_room) for r in self.rooms):
                self.rooms.append(new_room)
                self.add_room_to_grid(new_room)

    # Set grid values to 1 to indicate open space
    def add_room_to_grid(self, room):
        for i in range(room.x, room.x + room.width):
            for j in range(room.y, room.y + room.height):
                self.grid[j][i] = 1

    # Connects rooms
    def connect_rooms(self):
        if not self.rooms:
            return

        # Start with a random room
        start_room = random.choice(self.rooms)
        start_room.connected = True

        # Continue connecting
        while not all(room.connected for room in self.rooms):
            self.connect_next_room()

    def connect_next_room(self):
        unconnected_rooms = [room for room in self.rooms if not room.connected]
        connected_rooms = [room for room in self.rooms if room.connected]

        closest_pair = None
        closest_distance = float('inf')

        for room1 in unconnected_rooms:
            for room2 in connected_rooms:
                distance = self.distance_between_rooms(room1, room2)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_pair = (room1, room2)

        if closest_pair:
            self.create_hallway(*closest_pair)
            closest_pair[0].connected = True

    # Additional random connections
    def connect_randomly(self, max_connections):
        for room1 in self.rooms:

            num_connections = random.randint(0, max_connections)
            for _ in range(num_connections):
                room2 = random.choice(self.rooms)

                self.create_hallway(room1, room2)
                room1.connected = True

    def distance_between_rooms(self, room1, room2):
        x1, y1 = room1.center()
        x2, y2 = room2.center()
        return abs(x1 - x2) + abs(y1 - y2)

    def create_hallway(self, room1, room2):
        x1, y1 = room1.center()
        x2, y2 = room2.center()

        # Determine the order of hallway creation (horizontal then vertical or vice versa)
        horizontal_first = random.choice([True, False])

        # Create the horizontal segment
        for x in range(min(x1, x2), max(x1, x2) + 1):
            for w in range(-self.hallway_width // 2, self.hallway_width // 2 + 1):
                if horizontal_first:
                    self.grid[y1 + w][x] = 1
                else:
                    self.grid[y2 + w][x] = 1

        # Create the vertical segment
        for y in range(min(y1, y2), max(y1, y2) + 1):
            for w in range(-self.hallway_width // 2, self.hallway_width // 2 + 1):
                if horizontal_first:
                    self.grid[y][x2 + w] = 1
                else:
                    self.grid[y][x1 + w] = 1

        # Fill in the corner
        if horizontal_first:
            for w in range(-self.hallway_width // 2, self.hallway_width // 2 + 1):
                for h in range(-self.hallway_width // 2, self.hallway_width // 2 + 1):
                    self.grid[y1 + w][x2 + h] = 1
        else:
            for w in range(-self.hallway_width // 2, self.hallway_width // 2 + 1):
                for h in range(-self.hallway_width // 2, self.hallway_width // 2 + 1):
                    self.grid[y2 + w][x1 + h] = 1

    def print_grid(self):
        for row in self.grid:
            print(" ".join(str(cell) for cell in row))

    def get_obstacles(self):
        obstacles = []
        for y in range(self.height):
            for x in range(self.width):
                # If the grid value is 0 and it is adjacent to 1, it must be a wall
                if self.grid[y][x] == 0 and self.is_adjacent_to_one(x, y):
                    obstacles.append((x, y))
        return obstacles

    def is_adjacent_to_one(self, x, y):
        # Check all surrounding positions
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if 0 <= x + i < self.width and 0 <= y + j < self.height:
                    if self.grid[y + j][x + i] == 1:
                        return True
        return False

    # Used to scale the environment up or down. Down will usually result in information loss (hallways disappear)
    #   Generally best to scale up. That way a small environment can be easily generated before being scaled to
    #   something like pixel coordinates
    def scale_grid(self, new_width, new_height):
        scaled_grid = [[0 for _ in range(new_width)] for _ in range(new_height)]

        x_scale = new_width / self.width
        y_scale = new_height / self.height

        for y in range(new_height):
            for x in range(new_width):
                orig_x = int(x / x_scale)
                orig_y = int(y / y_scale)
                scaled_grid[y][x] = self.grid[orig_y][orig_x]

        self.starting_points = np.array(list(self.starting_points)) * np.array([x_scale, y_scale])
        self.r_radius *= x_scale

        self.width = new_width
        self.height = new_height
        self.grid = scaled_grid
        self.polygon, self.polygon_arr = self.convert_to_poly()
        self.quadtree = QuadTree(self.width, self.height)
        self.populate_quadtree()

    def draw_env(self, filename='env.png'):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width * 10, self.height * 10)
        ctx = cairo.Context(surface)

        # Draw the grid
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == 1:
                    ctx.rectangle(x * 10, y * 10, 10, 10)
                    ctx.set_source_rgb(1, 1, 1)
                else:
                    ctx.set_source_rgb(0, 0, 0)
                ctx.fill()

        surface.write_to_png(filename)

    # Easiest way to convert environment generated as a grid to a polygon is to use contour detection. Even for a
    # simple scenario this seems to have errors. However, these errors actually seem to contribute to the environment
    def convert_to_poly(self):
        array = np.array(self.grid, dtype=np.uint8) * 255
        if self.rand_connections > 0:
            contours, _ = cv2.findContours(array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Use external if not using random connects. Random connections can sometimes result in every room being merged
        #   into a single polygon. This merging behavior is desirable for small subsets of rooms
        else:
            contours, _ = cv2.findContours(array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_vertices = []
        for cnt in contours:
            epsilon = 0.00001 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            vertices = approx.reshape(-1, 2)
            vertices_list = [tuple(point) for point in vertices]

            contour_vertices.append(vertices_list)

            cv2.drawContours(array, [approx], 0, (0, 255, 0), 3)
        cv2.imwrite('contours.png', array)

        plt.figure()

        # Plot each contour's vertices
        for vertices in contour_vertices:
            xs, ys = zip(*vertices)
            xs = list(xs)
            ys = list(ys)
            first_x = xs[0]
            first_y = ys[0]
            xs.append(first_x)
            ys.append(first_y)
            plt.plot(xs, ys, c='b')

        xs, ys = zip(*self.starting_points)
        plt.scatter(xs, ys, )

        plt.title('Environment')

        plt.savefig('env_poly.png')

        poly_arr = []
        for polygon in contour_vertices:
            n = len(polygon)
            for i in range(n):
                line = [polygon[i], polygon[(i + 1) % n]]
                poly_arr.append(line)
        poly_arr = np.array(poly_arr)

        return contour_vertices, poly_arr

random.seed(313)
env = SimEnv(width=250, height=250, min_room_size=25, max_room_size=50, min_rooms=500, max_rooms=500, hallway_width=5,
            n_robots=5, r_radius=2, rand_connections=0)
env.print_grid()
env.draw_env('env.png')
obstacles = env.get_obstacles()
env.scale_grid(1000, 1000)
polygon = env.convert_to_poly()
