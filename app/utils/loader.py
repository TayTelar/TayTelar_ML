from ultralytics import YOLO
from app.utils.logger import logger

import numpy as np
import yaml
class Loader:

    @staticmethod
    def load_model(model_path="app/services/best.pt"):
        logger.info("Loading model from: %s" % model_path)
        model= YOLO(model_path)
        # dummy_image = np.zeros((1, 3, 640, 640))
        dummy_image = "C:/Users/ADMIN/Desktop/backup_ds/new_pant_ds -no aug/images/test/test-1.jpg"
        logger.info("Testing with dummy image")
        model(dummy_image)  # This will load the model and make a dummy prediction to ensure it's loaded correctly.'
        logger.info("Model loaded successfully")
        return model

    @staticmethod
    def load_garment_config():
        with open("app/services/properties.yaml", "r") as file:
            config = yaml.safe_load(file)
        return config