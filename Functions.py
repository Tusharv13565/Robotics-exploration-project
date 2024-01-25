from collections import deque

import cairo
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from PathGraph import update_graph


# Rotate a 2D vector by a given angle.
def rotate_vector(vector, angles):
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    rotation_matrix = np.array([
        [cos_angles, -sin_angles],
        [sin_angles, cos_angles]
    ])

    # Adjusting the shapes for broadcasting
    rotation_matrix = np.transpose(rotation_matrix, (2, 0, 1))
    vector = vector.reshape((2, 1))

    # Broadcasting the dot product over the angles
    return np.dot(rotation_matrix, vector).reshape(-1, 2)


def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# Finds the intersection point of two lines
def line_intersection(line1, line2):
    xdiff = np.array([line1[0][0] - line1[1][0], line2[0][0] - line2[1][0]])
    ydiff = np.array([line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None  # Lines don't intersect

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


# Check if point c is on line segment ab
def is_between(a, b, c):
    cross_product = (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1])
    if abs(cross_product) > 1e-7:
        return False

    dot_product = (c[0] - a[0]) * (b[0] - a[0]) + (c[1] - a[1]) * (b[1] - a[1])
    if dot_product < 0:
        return False

    squared_length = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
    if dot_product > squared_length:
        return False

    return True


def plot_detection(robot_position, perception_cone, polygons, intersections):
    for polygon in polygons:
        polygon_with_closure = polygon + [polygon[0]]
        x, y = zip(*polygon_with_closure)
        plt.plot(x, y, 'b-')

    # Plot the detection vectors
    for vector in perception_cone:
        end_point = robot_position + vector
        plt.arrow(robot_position[0], robot_position[1],
                  end_point[0] - robot_position[0], end_point[1] - robot_position[1],
                  head_width=0.05, head_length=0.1, fc='r', ec='r')

    # Plot the intersection points
    for point in intersections:
        plt.plot(point[0], point[1], 'go')

    # Mark the robot's position
    plt.plot(robot_position[0], robot_position[1], 'ko')

    plt.title('Robot Perception')
    plt.axis('equal')
    plt.show()


def plot_robot_paths(robots, polygons):
    plt.figure(figsize=(10, 10))

    for polygon in polygons:
        polygon_with_closure = polygon + [polygon[0]]
        x, y = zip(*polygon_with_closure)
        plt.plot(x, y, 'b-')

    for robot in robots:
        x_positions = [position[0] for position in robot.position_history]
        y_positions = [position[1] for position in robot.position_history]

        plt.plot(x_positions, y_positions, marker='o', markersize=1)

    plt.title('Robot Positions')
    plt.savefig('sim_output.png')


# Animates full path of robots
def animate_paths(robots, polygons):
    plt.clf()
    fig, ax = plt.subplots()

    # Lines for paths
    path_lines = [ax.plot([], [], label=f'Robot {i}')[0] for i, _ in enumerate(robots)]

    # Lines for orientation and velocity vectors
    orientation_lines = [ax.plot([], [], 'r-', linewidth=1)[0] for _ in robots]
    velocity_lines = [ax.plot([], [], 'g-', linewidth=1)[0] for _ in robots]

    # Plot polygons
    for polygon in polygons:
        polygon_with_closure = polygon + [polygon[0]]
        x, y = zip(*polygon_with_closure)
        plt.plot(x, y, 'b-')

    def init():
        return path_lines + orientation_lines + velocity_lines

    def update(frame):
        for i, robot in enumerate(robots):
            # Update path
            if frame < len(robot.position_history):
                x, y = zip(*robot.position_history[:frame + 1])
                path_lines[i].set_data(x, y)

            # Update orientation vector
            if frame < len(robot.orientation_history):
                ox, oy = robot.orientation_history[frame]
                pos_x, pos_y = robot.position_history[frame]
                orientation_lines[i].set_data([pos_x, pos_x + ox], [pos_y, pos_y + oy])

            # Update velocity vector
            if frame < len(robot.velocity_history):
                vx, vy = robot.velocity_history[frame]
                pos_x, pos_y = robot.position_history[frame]
                velocity_lines[i].set_data([pos_x, pos_x + vx], [pos_y, pos_y + vy])

        return path_lines + orientation_lines + velocity_lines

    anim = FuncAnimation(fig, update, frames=max(len(r.position_history) for r in robots),
                         init_func=init, blit=True)

    # plt.show()
    anim.save('paths.gif', fps=30)

    return anim


def animate_sim(robots, polygons, occupancy_grid):
    plt.clf()
    fig, ax = plt.subplots()

    # Scatter plot for robot positions
    scatters = [ax.scatter([], [], s=1, label=f'Robot {i}') for i, _ in enumerate(robots)]

    # Lines for orientation and velocity vectors
    orientation_lines = [ax.plot([], [], 'r-', linewidth=1)[0] for _ in robots]
    velocity_lines = [ax.plot([], [], 'g-', linewidth=1)[0] for _ in robots]

    # path_lines = [ax.plot([], [], 'b-', linewidth=1)[0] for _ in robots]

    cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'white'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    occupancy_img = ax.imshow(np.transpose(occupancy_grid[0]), cmap=cmap, norm=norm, interpolation='nearest', alpha=0.5,
                              origin='lower')

    # Plot polygons
    for polygon in polygons:
        polygon_with_closure = polygon + [polygon[0]]
        x, y = zip(*polygon_with_closure)
        plt.plot(x, y, 'b-')

    def init():
        return scatters + orientation_lines + velocity_lines

    def update(frame):
        if frame < len(occupancy_grid):
            occupancy_img.set_data(np.transpose(occupancy_grid[frame]))
        for i, robot in enumerate(robots):
            # Update robot positions
            if frame < len(robot.position_history):
                x, y = robot.position_history[frame]
                scatters[i].set_offsets([x, y])

            # Update orientation vector
            if frame < len(robot.orientation_history):
                ox, oy = robot.orientation_history[frame]
                pos_x, pos_y = robot.position_history[frame]
                orientation_lines[i].set_data([pos_x, pos_x + ox * 5], [pos_y, pos_y + oy * 5])

            # Update velocity vector
            if frame < len(robot.velocity_history):
                vx, vy = robot.velocity_history[frame]
                pos_x, pos_y = robot.position_history[frame]
                velocity_lines[i].set_data([pos_x, pos_x + vx * 5], [pos_y, pos_y + vy * 5])

            # if frame < len(robot.path_history):
            #     path = robot.path_history[frame]
            #     position = np.expand_dims(robot.position_history[frame], 0)
            #     path_history = np.concatenate([position, path])
            #     pos_x, pos_y = zip(*path_history)
            #     path_lines[i].set_data(pos_x, pos_y)

        return [occupancy_img] + scatters + orientation_lines + velocity_lines  # + path_lines

    anim = FuncAnimation(fig, update, frames=max(len(r.position_history) for r in robots),
                         init_func=init, blit=True)

    # plt.show()
    anim.save('sim_run.gif', fps=30)

    return anim


def plot_segment_with_frontier(segment_index, segments, frontier_targets):
    plt.clf()
    plt.cla()
    if segment_index >= len(segments) or segment_index >= len(frontier_targets):
        print("Invalid segment index")
        return

    segment = segments[segment_index]
    frontier_space = frontier_targets[segment_index]

    # Plotting the graph segment
    x, y = zip(*segment)
    plt.plot(x, y, marker='o', markersize=5, linestyle='-', color='blue', label='Segment')

    # Plotting the frontier space
    if len(frontier_space) > 0:
        fx, fy = zip(*frontier_space)
        plt.scatter(fx, fy, marker='x', color='red', label='Frontier Space')
    plt.legend()
    plt.savefig('segment_and_frontier.png')


def draw_frontier_grid(frontier_grid, occupancy_grid, filename='frontier_grid.png'):
    occupancy_grid = np.transpose(occupancy_grid)
    frontier_grid = np.transpose(frontier_grid)
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, occupancy_grid.shape[0] * 10, occupancy_grid.shape[1] * 10)
    ctx = cairo.Context(surface)

    # Calculate the height of the surface to flip the y-axis
    surface_height = occupancy_grid.shape[1] * 10

    # Draw the grid
    for x in range(occupancy_grid.shape[0]):
        for y in range(occupancy_grid.shape[1]):
            # Flip the y-axis
            flipped_y = surface_height - (y + 1) * 10

            # Draw the occupancy grid
            ctx.rectangle(x * 10, flipped_y, 10, 10)
            if occupancy_grid[y][x] == 0:
                ctx.set_source_rgb(0, 0, 0)  # Black for unoccupied
            elif occupancy_grid[y][x] == 1:
                ctx.set_source_rgb(1, 1, 1)  # White for occupied
            else:
                ctx.set_source_rgb(0.5, 0.5, 0.5)  # Grey for unknown

            # Draw the frontier grid
            if frontier_grid[y, x] == 1:
                ctx.set_source_rgb(1, 0, 0)  # Red for frontier

            ctx.fill()

    surface.write_to_png(filename)


def draw_occupancy(occupancy_grid, filename='occupancy.png'):
    occupancy_grid = np.flip(occupancy_grid, axis=1)
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, occupancy_grid.shape[0] * 10, occupancy_grid.shape[1] * 10)
    ctx = cairo.Context(surface)

    # Draw the grid
    for x in range(occupancy_grid.shape[0]):
        for y in range(occupancy_grid.shape[1]):
            if occupancy_grid[x][y] == 0:
                ctx.rectangle(x * 10, y * 10, 10, 10)
                ctx.set_source_rgb(0, 0, 0)
            elif occupancy_grid[x][y] == 1:
                ctx.rectangle(x * 10, y * 10, 10, 10)
                ctx.set_source_rgb(1, 1, 1)
            else:
                ctx.rectangle(x * 10, y * 10, 10, 10)
                ctx.set_source_rgb(0.5, 0.5, 0.5)

            ctx.fill()

    surface.write_to_png(filename)
