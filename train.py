import ultralytics
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
def main():
    global model
    torch.cuda.empty_cache()

    try:
        model = YOLO() #"yolov8m-pose.pt"
    except RuntimeError as e:
        print(f"Failed to load model: {e}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Train the model
    results = model.train(data='config.yaml',
                          epochs=100,
                          imgsz=640,
                          batch=10,
                          device="cpu"
                          )
    model_path = "trained_phone_keypoint_m.pt"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    results.val()

if __name__ == '__main__':
    main()
