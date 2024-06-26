import numpy as np


def format_point(point, scale=1e3):  # Normalizing function, returns tuple
    return round(point[0] * scale) / scale, round(point[1] * scale) / scale


def format_point2(point, scale=1e3):  # Normalizing function, returns dictionary
    return {"x": round(point[0] * scale) / scale, "y": round(point[1] * scale) / scale}


def is_close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def transform_point_to_tuple(x, y, matrix):
    point = np.dot(matrix, [x, y, 1])
    return point[0], point[1]


def transform_point_to_list(x, y, matrix):
    # End goal fix would be to not call this function if matrix is np.identity(3) and this must be done without
    # if checks. Basically, the system should be configured as such that it is always known whether the matrix is
    # identity or not.
    if matrix is np.identity(3):
        return [x, y]
    point = np.dot(matrix, [x, y, 1])
    return [point[0], point[1]]


def transform_height(y, matrix):
    point = np.dot(matrix, [0, y, 0])
    return point[1]


def transform_points(matrix, points):
    return [transform_point_to_tuple(p[0], p[1], matrix) for p in points]


def transform_points_vectorized(matrix, points):
    # Convert points to a NumPy array and add a column of ones for homogeneous coordinates
    points_array = np.hstack((np.array(points), np.ones((len(points), 1))))
    # Perform matrix multiplication
    transformed_points = points_array @ matrix.T
    # Return the transformed points, ignoring the homogeneous coordinate
    return transformed_points[:, :2].tolist()


def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v


def compute_rotation_matrix(normal, target):
    """
    Compute the rotation matrix that rotates 'normal' to 'target'.
    'normal' and 'target' must be normalized.
    """
    v = np.cross(normal, target)
    c = np.dot(normal, target)
    k = 1 / (1 + c)

    vx, vy, vz = v
    return np.array([
        [vx*vx*k + c,   vx*vy*k - vz,  vx*vz*k + vy],
        [vy*vx*k + vz,  vy*vy*k + c,   vy*vz*k - vx],
        [vz*vx*k - vy,  vz*vy*k + vx,  vz*vz*k + c  ]
    ])
