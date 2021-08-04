#!/usr/bin/env python

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de
# MSR Project Sem 2

# Utility functions

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import numpy as np
from math import pi
from typing import Tuple, NoReturn

def wrapToPi(theta: float) -> (float):
    """
    Wrap around angles to the range [-pi, pi]
    Args:
        - theta: angle
    Returns:
        - theta: angle within range [-pi, pi]
    """
    while theta < -pi:
        theta = theta + 2 * pi
    while theta > pi:
        theta = theta - 2 * pi
    return theta

def wrapTo2Pi(theta: float) -> (float):
    """
    Wrap around angles to the range [0, 2pi]
    Args:
        - theta: angle
    Returns:
        - theta: angle within range [0, 2pi]
    """
    while theta < 0:
        theta = theta + 2 * pi
    while theta > 2 * pi:
        theta = theta - 2 * pi
    return theta

def euclidean_distance(pt_A: np.ndarray, pt_B: np.ndarray) -> (float):
    """
    Calculate Euclidean distance between two points A and B in 2D
    Args:
        - pt_A: (x, y) coordinate of point A
        - pt_B: (x, y) coordinate of point B
    Returns:
        - dist: distance between points A and B
    """
    dist = np.linalg.norm(pt_A - pt_B)
    return dist

def read_file(path: str, delimiter: str = ' ') -> (np.ndarray):
    """ Read data from a file

    Arguments
    ---------
    - path: Path of ASCII file to read
    - delimiter: Delimiter character for the file to be read
    
    Returns
    -------
    - data: Data from file as a numpy array
    """
    data = np.loadtxt(path, delimiter=delimiter)
    return data