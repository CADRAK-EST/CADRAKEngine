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
