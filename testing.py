import numpy as np
import timeit

# Original transform_point function
def transform_point(x, y, matrix):
    point = np.dot(matrix, [x, y, 1])
    return point[0], point[1]

def transform_points_original(matrix, points):
    return [transform_point(p[0], p[1], matrix) for p in points]

# Refactored transform_points function
def transform_points_vectorized(matrix, points):
    points_array = np.array(points)
    ones_column = np.ones((points_array.shape[0], 1))
    homogeneous_points = np.hstack((points_array, ones_column))
    transformed_points = homogeneous_points @ matrix.T
    return transformed_points[:, :2].tolist()

# Generate a large set of random points
num_points = 100000
points = [(np.random.rand(), np.random.rand()) for _ in range(num_points)]

# Identity transformation matrix
identity_matrix = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])

# Test the performance of the original function
original_time = timeit.timeit(lambda: transform_points_original(identity_matrix, points), number=10)

# Test the performance of the refactored vectorized function
vectorized_time = timeit.timeit(lambda: transform_points_vectorized(identity_matrix, points), number=10)

print(f"Original function time: {original_time:.6f} seconds")
print(f"Vectorized function time: {vectorized_time:.6f} seconds")
