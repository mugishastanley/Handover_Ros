import numpy as np

def vector_distance_and_normalized_direction(x1, y1, z1, x2, y2, z2):
    # Create NumPy arrays from the input coordinates
    vector1 = np.array([x1, y1, z1])
    vector2 = np.array([x2, y2, z2])

    # Calculate the Euclidean distance between the two vectors
    distance = np.linalg.norm(vector2 - vector1)

    # Calculate the direction vector
    direction_vector = vector2 - vector1

    # Normalize the direction vector
    normalized_direction = direction_vector / np.linalg.norm(direction_vector)

    return distance, normalized_direction

# Example usage:
x1, y1, z1 = 1, 2, 3
x2, y2, z2 = 4, 5, 6

distance, normalized_direction = vector_distance_and_normalized_direction(x1, y1, z1, x2, y2, z2)

print(f"Distance between vectors: {distance}")
print(f"Normalized direction vector: {normalized_direction}")
