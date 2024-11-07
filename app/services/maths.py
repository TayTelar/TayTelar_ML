import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class Maths:
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

        # Visualize results
        plt.figure(figsize=(8, 6))
        plt.scatter(points[:, 0], points[:, 1], color="red", label="Points", zorder=5)
        plt.plot(points[:, 0], points[:, 1], 'b-', zorder=4)

        is_collinear = np.all(angles <= angle_threshold)
        title = "Points are" + (" approximately" if is_collinear else " NOT approximately") + " on a straight line"
        plt.title(title)
        plt.show()

        # Print detailed analysis if needed
        if angles.size > 0:
            for i, angle in enumerate(angles, 1):
                print(f"Angle {i}: {angle:.2f} degrees")

        print(title + ".")
        return is_collinear


if __name__ == "__main__":
    # Test cases
    collinear_points = [[808.2897338867188, 443.1014099121094],
                        [768.1558837890625, 703.1568603515625],
                        [755.0282592773438, 808.4175415039062],
                        [657.451416015625, 1810.53759765625]]

    Maths.is_approximately_on_straight_line(collinear_points)