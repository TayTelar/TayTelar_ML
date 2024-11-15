import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class Calc:
    @staticmethod
    def is_approximately_on_straight_line(points: List[List[float]], angle_threshold: float = 10) -> bool:
        """
        Check if points are approximately collinear using vector analysis.

        Args:
            points: List of [x, y] coordinates
            angle_threshold: Maximum allowed angle deviation in degrees (default: 10)

        Returns:
            bool: True if points are approximately collinear, False otherwise

        Raises:
            ValueError: If points is None or has less than 2 points
        """
        if points is None or len(points) < 2:
            raise ValueError("Insufficient points to check linearity")

        # Convert to numpy array for better performance
        points = np.array(points)

        # Calculate vectors between consecutive points in one operation
        vectors = points[1:] - points[:-1]

        # Calculate vector magnitudes
        magnitudes = np.linalg.norm(vectors, axis=1)

        # Calculate dot products between consecutive vectors
        dot_products = np.sum(vectors[:-1] * vectors[1:], axis=1)

        # Calculate cosine of angles
        cos_angles = dot_products / (magnitudes[:-1] * magnitudes[1:])

        # Handle numerical precision issues
        cos_angles = np.clip(cos_angles, -1, 1)

        # Calculate angles in degrees
        angles = np.degrees(np.arccos(cos_angles))

        is_collinear = np.all(angles <= angle_threshold)

        # Print detailed analysis if needed
        # if angles.size > 0:
        #     for i, angle in enumerate(angles, 1):
        #         print(f"Angle {i}: {angle:.2f} degrees")

        return is_collinear
    @staticmethod
    def calculate_total_distance(keypoints: List[List[float]], is_circular_distance: bool = False) -> float:
        """
        Calculate the total Euclidean distance between consecutive points using vectorized operations.

        Args:
            keypoints: List of points, where each point is [x, y]
            is_circular_distance: If True, includes distance from last point back to first point

        Returns:
            float: Total distance between points

        Example:
            >>> points = [[0, 0], [3, 4], [6, 8]]
            >>> calculate_total_distance(points)
            10.0  # 5.0 (first segment) + 5.0 (second segment)
            >>> calculate_total_distance(points, is_circular_distance=True)
            18.0  # 10.0 + 8.0 (distance back to start)
        """
        if not keypoints or len(keypoints) < 2:
            return 0.0

        # Convert to numpy array for vectorized operations
        points = np.array(keypoints)

        # Calculate differences between consecutive points
        diff = points[1:] - points[:-1]

        # Calculate Euclidean distances in one operation
        distances = np.sqrt(np.sum(diff ** 2, axis=1))

        # Sum all distances
        total = float(np.sum(distances))

        # Add distance back to start if circular
        if is_circular_distance:
            circular_distance = np.sqrt(np.sum((points[0] - points[-1]) ** 2))
            total += float(circular_distance)

        return total


if __name__ == "__main__":
    # Test cases
    collinear_points = [[808.2897338867188, 443.1014099121094],
                        [768.1558837890625, 703.1568603515625],
                        [755.0282592773438, 808.4175415039062],
                        [657.451416015625, 1810.53759765625]]

    Calc.is_approximately_on_straight_line(collinear_points)