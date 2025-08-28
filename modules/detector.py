import torch
from ultralytics import YOLO

class YOLOv8FaceDetector:
    def __init__(self, model_path, conf_thres=0.5, use_gpu=True):
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.device = device
        self.conf_thres = conf_thres
        self.model.to(device)

    def detect(self, frame):
        results = self.model.predict(frame, device=self.device, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > self.conf_thres:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append([int(x1), int(y1), int(x2), int(y2)])
        return detections
