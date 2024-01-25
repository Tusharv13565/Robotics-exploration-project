import numpy as np


def steer_behavior(robot, target):
    desired_direction = target - robot.position

    steering_force = desired_direction - robot.velocity
    return steering_force
