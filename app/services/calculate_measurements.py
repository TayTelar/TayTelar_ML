from typing import List, Dict
import cv2

from app.utils.calc import Calc
from app.utils.loader import Loader
import numpy as np

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


def calculate(ratio:float,image:np.ndarray):
    kp_predictions = CalcPantMeasures.import_image_and_predict(image)
    return CalcPantMeasures.calculations(kp_predictions,ratio)


class CalcPantMeasures:

    @staticmethod
    def import_image_and_predict(req_image):
        image = cv2.imread(req_image)  # Load the request image
        if image is None:
            raise FileNotFoundError(f"Error: Unable to load the image at {req_image}")
        res = model(image)
        return res[0]

    @staticmethod
    def calc_required_measures(
            keypoints: List[List[float]],
            pant_properties: Dict[str, Dict],
            required_measures: List[str],
            ratio: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate measurements based on keypoints and properties.

        Args:
            keypoints: List of [x, y] coordinates
            pant_properties: Dictionary containing measurement configurations
            required_measures: List of measurement names to calculate
            ratio: Conversion ratio (units/pixel)

        Returns:
            Dictionary of calculated measurements
        """
        props = {}

        if ratio < 0:
            raise ValueError("Ratio cannot be negative")

        for measure in required_measures:
            measurement_config = pant_properties.get(measure)
            if measurement_config:
                indices = measurement_config["indices"]
                is_circular_distance = measurement_config["is_circular_distance"]

                try:
                    extracted_kp = extract_indices(keypoints, indices)
                    distance = Calc.calculate_total_distance(extracted_kp, is_circular_distance)

                    if ratio != 0.0:
                        # Convert from pixels to actual units
                        distance = round(distance * ratio, 2)

                    props[measure] = distance
                    print(f"{measure} -> {distance} {'units' if ratio != 0 else 'pixels'}")
                except Exception as e:
                    print(f"Error calculating {measure}: {str(e)}")
                    continue
            else:
                print(f"Measurement {measure} not found in pant_properties")

        return props

    @staticmethod
    def calculations(predictions,ratio=0.0):
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
                    is_collinear = Calc.is_approximately_on_straight_line(right_outseam_kp)
                    if is_collinear:
                        # If the points are collinear, calculate the required measurements
                        print("Pant placed correctly")
                        return CalcPantMeasures.calc_required_measures(key_points, pant_properties, required_measures,ratio)

                    else:
                        print("Please keep the right leg straight")
            else:
                print("Pant properties or required measures not found")

        else:
            print("Object not detected")


if __name__ == "__main__":
    request_image_path = "C:/Users/ADMIN/Desktop/backup_ds/new_pant_ds -no aug/images/test/test-5.jpg"
    try:
        predictions = CalcPantMeasures.import_image_and_predict(request_image_path)
        CalcPantMeasures.calculations(predictions)

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except ValueError as v_error:
        print(v_error)
    except Exception as e:
        print(e)
