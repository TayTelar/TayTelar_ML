from ultralytics import YOLO
import yaml
class Loader:

    @staticmethod
    def load_model(model_path="C:/Users/ADMIN/Downloads/taytelar-ml - python/app/static/model/yolov8-best_1.pt"):
        return YOLO(model_path)

    @staticmethod
    def load_garment_config():
        with open("properties.yaml", "r") as file:
            config = yaml.safe_load(file)
        return config