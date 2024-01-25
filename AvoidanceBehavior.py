import numpy as np


def avoidance_behavior(robot, intersections):
    intersections = np.array(intersections)

    # Calculate vectors from intersections to robot
    if len(intersections) == 0:
        avoidance_vector = np.array([0, 0])
    else:
        to_robot_vectors = robot.position - intersections

        # Calculate distances and normalize vectors
        distances = np.linalg.norm(to_robot_vectors, axis=1)
        normalized_vectors = np.divide(to_robot_vectors.T, distances, out=np.zeros_like(to_robot_vectors.T), where=distances!=0).T

        # Inversely weigh by distance and sum
        within_range_mask = (distances > 0) & (distances <= robot.avoid_range)
        avoidance_vector = np.sum(normalized_vectors[within_range_mask] / distances[within_range_mask, None], axis=0)

        # Normalize the final avoidance vector
        norm_avoidance_vector = np.linalg.norm(avoidance_vector)
        if norm_avoidance_vector > 0:
            avoidance_vector /= norm_avoidance_vector

    return avoidance_vector
