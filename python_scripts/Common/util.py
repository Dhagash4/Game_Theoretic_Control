#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de

# MSR Project Sem 2: Game Theoretic Control for Multi-Car Racing

# This script defines some utility functions that are used often in other scripts

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import numpy as np
from math import pi
import _pickle as cpickle
from typing import NoReturn

def wrapToPi(theta: float) -> (float):
    """
    Wrap around angles to the range [-pi, pi]

    Arguments
    ---------
    - theta: angle

    Returns
    -------
    - theta: angle within range [-pi, pi]
    """
    while theta < -pi:
        theta += 2 * pi
    while theta > pi:
        theta -= 2 * pi
    return theta

def wrapTo2Pi(theta: float) -> (float):
    """
    Wrap around angles to the range [0, 2pi]
    
    Arguments
    ---------
    - theta: angle

    Returns
    -------
    - theta: angle within range [0, 2pi]
    """
    while theta < 0:
        theta += 2 * pi
    while theta > 2 * pi:
        theta -= 2 * pi
    return theta

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

def read_pickle(path: str) -> (np.ndarray):
    """ Read data from a pickle file

    Arguments
    ---------
    - path: Path of the pickle file to read
    
    Returns
    -------
    - data: Data from file as a numpy array
    """
    data = []

    with open(path, 'rb') as f:
        while True:
            try:
                data.append(cpickle.load(f))
            except EOFError:
                break
    return np.array(data)