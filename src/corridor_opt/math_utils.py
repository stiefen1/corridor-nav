import numpy as np

def normalize_angle_0_2pi(angle):
    """
    Map any angle in radians to the interval [0, 2π]
    
    Args:
        angle: Angle in radians (can be any real number)
    
    Returns:
        Normalized angle in [0, 2π]
    """
    return angle % (2 * np.pi)