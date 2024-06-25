import time
from shapely.geometry import MultiPoint

# Sample points data
points = [(i, i) for i in range(10000)]

# Measure the time taken for multipoint creation
start_time = time.time()
for _ in range(100):
    multipoint = MultiPoint(points)
end_time = time.time()

print(f"Multipoint creation took {end_time - start_time:.4f} seconds")
