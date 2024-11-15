from typing import List, Dict, Tuple, Optional, Any
from flask import current_app
from app.utils.calc import Calc
from app.utils.logger import logger

# Constants
MM_TO_INCH = 0.393701

class Helper:
    """Helper class containing static utility methods for measurement calculations"""

    @staticmethod
    def get_predictions(image: str) -> Any:
        """
        Get predictions from the ML model for the given image.

        Args:
            image: Input image for prediction

        Returns:
            Model prediction results or None if no predictions found
        """
        model = current_app.config['MODEL']
        results = model(image)
        if results is None:
            logger.error("No predictions found")
            return None
        return results[0]

    @staticmethod
    def get_garment_configuration(obj: str) -> Tuple[Optional[Dict], Optional[List], Optional[Dict]]:
        """
        Retrieve garment configuration from app config.

        Args:
            obj: Garment type identifier

        Returns:
            Tuple containing keypoints config, required measurements, and validation measures
        """
        config_properties = current_app.config['CONFIG_PROPERTIES']
        config_data = config_properties.get('garments', {})
        garment_config = config_data.get(obj)

        if garment_config:
            logger.info(f"Found configuration for {obj}")
            return (
                garment_config.get('keypoints_config'),
                garment_config.get('required_measurements'),
                garment_config.get('keep_straight_measurements')
            )
        else:
            logger.error(f"{obj} configuration not found!")
            return None, None, None

    @staticmethod
    def extract_indices(data: List[List[float]], indices: List[int]) -> List[List[float]]:
        """
        Extract specific indices from a list of coordinates.

        Args:
            data: List of coordinate pairs
            indices: List of indices to extract

        Returns:
            List of extracted coordinates
        """
        extracted_data = []
        for index in indices:
            if 0 <= index < len(data):
                extracted_data.append(data[index])
            else:
                logger.warning(f"Index {index} out of range.")
        return extracted_data

    @staticmethod
    def verify_object_posture(
            predicted_keypoints: List[List[float]],
            keypoints_config: Dict,
            validation_measures: Dict
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify if the object's keypoints conform to expected posture.

        Args:
            predicted_keypoints: List of detected keypoint coordinates
            keypoints_config: Configuration for keypoint validation
            validation_measures: Validation measures for posture verification

        Returns:
            Tuple of (is_valid, error_measure_name)
        """
        for measure_name, validation_measure in validation_measures.items():
            angle_threshold = validation_measure.get('angle_threshold', 10.0)
            measure_indices = keypoints_config[measure_name]["indices"]
            extracted_kp = Helper.extract_indices(predicted_keypoints, measure_indices)

            if not Calc.is_approximately_on_straight_line(extracted_kp, angle_threshold):
                return False, measure_name

        return True, None

    @staticmethod
    def calc_required_measures(
            keypoints: List[List[float]],
            pant_properties: Dict[str, Dict],
            required_measures: List[str],
            ratio: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate required measurements based on keypoints.

        Args:
            keypoints: List of keypoint coordinates
            pant_properties: Configuration for measurements
            required_measures: List of measurements to calculate
            ratio: Conversion ratio from pixels to units

        Returns:
            Dictionary of calculated measurements
        """
        if ratio < 0:
            raise ValueError("Ratio cannot be negative")

        props = {"distance_unit": "inch"}

        for measure in required_measures:
            measurement_config = pant_properties.get(measure)
            if measurement_config:
                indices = measurement_config["indices"]
                is_circular_distance = measurement_config["is_circular_distance"]

                try:
                    extracted_kp = Helper.extract_indices(keypoints, indices)
                    distance = Calc.calculate_total_distance(extracted_kp, is_circular_distance)

                    if ratio != 0.0:
                        distance = round((distance * ratio) * MM_TO_INCH, 2)
                    props[measure] = distance
                except Exception as e:
                    logger.error(f"Error calculating {measure}: {str(e)}")
                    continue
            else:
                logger.warning(f"Measurement {measure} not found in pant_properties")

        return props


class CalculateMeasurement:
    """Main class for handling garment measurements"""

    def __init__(self):
        """Initialize calculation parameters from config"""
        properties = current_app.config['CONFIG_PROPERTIES']
        self._min_cls_conf = properties.get('min_cls_conf', 0.0)
        self._min_kp_conf = properties.get('min_kp_conf', 0.0)
        self._detected_obj = None

    def set_detected_obj(self, obj:Optional[str]) -> None:
        self._detected_obj = obj

    def verify_conf_of_key_points(self, kp:List[float])->None:
        smallest_kp_conf,corr_index = min((value , index) for index, value in enumerate(kp))
        if smallest_kp_conf < self._min_kp_conf :
            logger.error(
                f"Keypoint confidence {smallest_kp_conf} below threshold at point {corr_index + 1}")
            raise ValueError(f"Please check the {self._detected_obj} placed correctly or not!")

    def calculate(self, ratio: float, image: str) -> Tuple[bool, Dict]:
        """
        Calculate measurements for a garment image.

        Args:
            ratio: Conversion ratio from pixels to units
            image: Input image for measurement

        Returns:
            Tuple of (success, result_dict)
        """
        confidence_score = 0.0

        try:
            logger.info("Started calculating measurements")
            predictions = Helper.get_predictions(image)

            if predictions is None:
                raise ValueError("No predictions")

            self.set_detected_obj(predictions.names[0])
            bbox_conf_tensor = predictions.boxes.conf
            no_of_detected_objects = predictions.boxes.cls.numel()

            logger.info(f"Number of detected objects: {no_of_detected_objects}")

            if bbox_conf_tensor is None or bbox_conf_tensor.numel() == 0:
                confidence_score = 0.0
            else:
                confidence_score = float(predictions.boxes.conf[0].item())

            logger.info(f"Class confidence: {confidence_score}")

            if confidence_score < self._min_cls_conf:
                self.set_detected_obj(None)
                logger.info(f"Confidence score {confidence_score} is below threshold {self._min_cls_conf}")
                raise ValueError("Object is not detected")

            logger.info(f"Predicted object: {self._detected_obj}")

            if no_of_detected_objects != 1:
                raise ValueError(f"Multiple {self._detected_obj} detected")

            keypoints_config, required_measurements, validation_measures = Helper.get_garment_configuration(
                self._detected_obj)

            if not all([keypoints_config, required_measurements, validation_measures]):
                raise ValueError(f"No configuration found for the object {self._detected_obj}")

            predicted_keypoints_tensor = predictions.keypoints

            if predicted_keypoints_tensor is not None and len(predicted_keypoints_tensor) > 0:
                conf_list = predicted_keypoints_tensor.conf[0].tolist()
                self.verify_conf_of_key_points(conf_list)
                key_points = predicted_keypoints_tensor.xy[0].tolist()
                is_valid, failed_measure = Helper.verify_object_posture(key_points, keypoints_config,
                                                                        validation_measures)

                if not is_valid:
                    logger.error(f"Posture validation failed for {failed_measure}")
                    raise ValueError(f"Posture validation failed for {failed_measure}")

                result = Helper.calc_required_measures(key_points, keypoints_config, required_measurements, ratio)

                return True, {
                    "detected_object": self._detected_obj,
                    "confidence_score": confidence_score,
                    "measurements": result,
                    "ratio_used": ratio
                }


        except ValueError as ve:
            return False, {
                "detected_object": self._detected_obj,
                "errors": str(ve)
            }
        except Exception as e:
            logger.error(f"Error calculating measurements: {str(e)}")
            return False, {
                "detected_object": self._detected_obj,
                "errors": str(e)
            }