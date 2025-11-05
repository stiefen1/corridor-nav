import numpy as np

def rotation_matrix(angle) -> np.ndarray:
    return np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

def normalize_angle_0_2pi(angle):
    """
    Map any angle in radians to the interval [0, 2π]
    
    Args:
        angle: Angle in radians (can be any real number)
    
    Returns:
        Normalized angle in [0, 2π]
    """
    return angle % (2 * np.pi)

def normalize_angle_0_pi(angle):
    """
    Map any angle in radians to the interval [0, π]
    
    Args:
        angle: Angle in radians (can be any real number)
    
    Returns:
        Normalized angle in [0, π]
    """
    return angle % (np.pi)