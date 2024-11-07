from typing import List
from ultralytics import YOLO
import cv2
from maths import Maths
from loader import Loader
import numpy as np
import math

# Load necessary data
model = Loader.load_model()
config_properties = Loader.load_garment_config()

# Get garment config from the properties
def get_garment_properties(garment_name: str, config_data: dict):
    garments = config_data.get('garments', {})
    garment_config = garments.get(garment_name)

    if garment_config:
        return garment_config.get('keypoints_config'), garment_config.get('required_measurements')
    else:
        print(f"{garment_name} configuration not found!")
        return None, None

def extract_indices(data: List[List[float]], indices: List[int]) -> List[List[float]]:
    extracted_data = []
    for index in indices:
        # Check if the index is valid
        if 0 <= index < len(data):
            extracted_data.append(data[index])
        else:
            print(f"Warning: Index {index} out of range.")
    return extracted_data


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

class CalcPantMeasures:

    @staticmethod
    def import_image(req_image):
        image = cv2.imread(req_image)  # Load the request image
        if image is None:
            raise FileNotFoundError(f"Error: Unable to load the image at {req_image}")
        res = model(image)
        return res[0]

    @staticmethod
    def calc_required_measures(keypoints: List[List[float]], pant_properties: dict, required_measures: List[str]):
        for m in required_measures:
            # Check if the measurement exists in pant_properties
            measurement_config = pant_properties.get(m)
            if measurement_config:
                indices = measurement_config["indices"]
                is_circular_distance = measurement_config["is_circular_distance"]
                extracted_kp = extract_indices(keypoints, indices)
                distance=calculate_total_distance(extracted_kp,is_circular_distance)
                print(f"{m} -> {distance} pixels")

            else:
                print(f"Measurement {m} not found in pant_properties")

    @staticmethod
    def calculations(predictions):
        if predictions is None:
            raise ValueError("Predictions are empty")

        detected_object = predictions.names[0]

        if detected_object == "pant":

            key_points_tensor = predictions.keypoints

            # Get properties from the config file
            pant_properties, required_measures = get_garment_properties(detected_object, config_properties)

            print(f"req : {required_measures}")

            if pant_properties and required_measures:
                # Get right_outseam properties
                r_outseam = pant_properties.get("right_outseam")

                if key_points_tensor is not None and len(key_points_tensor) > 0:
                    key_points = key_points_tensor.xy[0].tolist()  # Extract the key points from the tensor

                    # Extract the right outseam keypoints
                    right_outseam_kp = extract_indices(key_points, r_outseam["indices"])

                    # Check if the points are collinear (i.e., leg is straight)
                    is_collinear = Maths.is_approximately_on_straight_line(right_outseam_kp)

                    if is_collinear:
                        # If the points are collinear, calculate the required measurements
                        CalcPantMeasures.calc_required_measures(key_points, pant_properties, required_measures)
                        print("Pant placed correctly")
                    else:
                        print("Please keep the right leg straight")
            else:
                print("Pant properties or required measures not found")

        else:
            print("Object not detected")

if __name__ == "__main__":
    request_image_path = "C:/Users/ADMIN/Desktop/backup_ds/new_pant_ds -no aug/images/test/test-34.jpg"
    try:
        predictions = CalcPantMeasures.import_image(request_image_path)
        CalcPantMeasures.calculations(predictions)

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except ValueError as v_error:
        print(v_error)
    except Exception as e:
        print(e)
