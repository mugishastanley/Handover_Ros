class MyClass:
    def points(self, x, y, z):
        """
        Returns a predicted point.
        Input: x, y, z
        Output: point 
        Algo:
            Define a pointset containing 3d points A, B, C, D, E
            Define a tol
            for all points in pointset
                if dist between a point and x, y, z is less than tol
                    return the point
            return None
        """
        # Define your pointset containing 3d points A, B, C, D, E
        pointset = [(0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.3, 0.3, 0.3), (0.4, 0.4, 0.4), (0.5, 0.5, 0.5)]
        tol = 0.2  # Define your tolerance
        
        closest_point_index = None
        min_dist = float('inf')
        
        # Calculate distance and find the closest point
        for i, point in enumerate(pointset):
            dist = ((point[0] - x) ** 2 + (point[1] - y) ** 2 + (point[2] - z) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest_point_index = i
        
        # Check if the closest point is within tolerance and return its index
        if min_dist < tol:
            return closest_point_index
        else:
            return None

# Example usage:
my_instance = MyClass()
x, y, z = 0.2, 0.2, 0.3  # Example values
predicted_point = my_instance.points(x, y, z)
print("Predicted Point:", predicted_point)
