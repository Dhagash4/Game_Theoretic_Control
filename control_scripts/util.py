# Author: Saurabh Gupta
# email: s7sagupt@uni-bonn.de
# MSR Project Sem 2

# Utility functions

import numpy as np

def wrapToPi(theta: float):
    """
    Wrap around angles to the range [-pi, pi]
    Args:
        - theta (float): angle
    Returns:
        - theta (float): angle within range [-pi, pi]
    """
    while theta < -math.pi:
        theta = theta + 2 * math.pi
    while theta > math.pi:
        theta = theta - 2 * math.pi
    return theta

def wrapTo2Pi(theta: float):
    """
    Wrap around angles to the range [0, 2pi]
    Args:
        - theta (float): angle
    Returns:
        - theta (float): angle within range [0, 2pi]
    """
    while theta < 0:
        theta = theta + 2 * math.pi
    while theta > 2 * math.pi:
        theta = theta - 2 * math.pi
    return theta

def euclidean_distance(pt_A: [float, float], pt_B: [float, float]):
    """
    Calculate Euclidean distance between two points A and B in 2D
    Args:
        - pt_A (float, float): (x, y) coordinate of point A
        - pt_B (float, float): (x, y) coordinate of point B
    Returns:
        - dist (float): distance between points A and B
    """
    dist = np.linalg.norm(pt_A - pt_B, ord = 'fro')
    return dist