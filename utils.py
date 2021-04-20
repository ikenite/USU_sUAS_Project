import numpy as np
from mat import mat


def in_half_plane(p, r, n):
    """
    in_half_plane returns a flag indicating if the position p is in the half
    plane defined by the point r and unit normal n.

    Inputs:
    p = position of the MAV (m)
    r = point on the half plane (m)
    n = unit normal of the half plane

    Outputs:
    flag indicating that p is in the half plane

    Example Usage
    in = in_half_plane( p,r,n )

    Determine whether p is in the half plane be calculating the sign of the
    dot product.
    """
    if ((p - r).T * n) >= 0:
        return 1
    else:
        return 0


def s_norm(x, y):
    """
    mag normalizes the sum of two vectors.
    This function is meant to make cleaner code since this case comes up
    frequently in the followWpp function.
    """
    return (x + y) / np.linalg.norm(x + y)


def Rz(theta):
    """
    return rotation vector about the z-axis
    """
    c = float(np.cos(theta))
    s = float(np.sin(theta))

    return mat([[c, -s, 0],
                [s, c, 0],
                [0, 0, 1]])


def i2p(chi_q):
    """
    summary of function goes here
    """
    c = np.cos(chi_q)
    s = np.sin(chi_q)

    return mat([[c, s, 0],
                [-s, c, 0],
                [0, 0, 1]])


def angle(v):
    """
    returns atan2
    """
    return np.arctan2(v[1], v[0])


def col(row_vector):
    """ returns a column vector from row_vector """
    if isinstance(row_vector, list):
        row_vector = np.array(row_vector)
    try:
        m, n = row_vector.shape
    except ValueError:
        return np.array([row_vector]).T
    if m == 1:
        return np.array(row_vector).T
    elif n == 1:
        return row_vector
    else:
        raise Exception("Invalid vector dimensions")
