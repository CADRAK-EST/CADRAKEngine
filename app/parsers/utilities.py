import numpy as np

def normalize_point(point, scale=1e3):  # Normalizing function
    return round(point[0] * scale) / scale, round(point[1] * scale) / scale


def is_close(a, b, tol=1e-9):
    return abs(a - b) <= tol


#def are_points_equal(p1, p2, tol=1e-9):
#    return is_close(p1[0], p2[0], tol) and is_close(p1[1], p2[1], tol)

def normalize_point2(point, scale=1e3):  # Normalizing function
    return {"x": round(point[0] * scale) / scale, "y": round(point[1] * scale) / scale}


def map_color(color, background_color):
    color_mapping = {
        1: '0xFF0000', 2: '0xFFFF00', 3: '0x00FF00', 4: '0x00FFFF', 5: '0x0000FF', 6: '0xFF00FF',
        7: '0xFFFFFF' if background_color.lower() == "0x000000" else '0x000000', 8: '0x808080', 9: '0xC0C0C0'
    }
    return color_mapping.get(color, '0x000000')

def transform_point(x, y, matrix):
    point = np.dot(matrix, [x, y, 1])
    return point[0], point[1]