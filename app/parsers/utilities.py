import numpy as np


def normalize_point(point, scale=1e3):  # Normalizing function
    return round(point[0] * scale) / scale, round(point[1] * scale) / scale


def is_close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def normalize_point2(point, scale=1e3):  # Normalizing function
    return {"x": round(point[0] * scale) / scale, "y": round(point[1] * scale) / scale}


def transform_point(x, y, matrix):
    point = np.dot(matrix, [x, y, 1])
    return point[0], point[1]


def apply_transform(matrix, points):
    return [transform_point(p[0], p[1], matrix) for p in points]
