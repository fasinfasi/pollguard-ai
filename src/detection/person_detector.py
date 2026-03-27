
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)

        detections = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id == 0:
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    detections.append({
                        "bbox": (x1,y1,x2,y2),
                        "confidence": conf
                    })

        return detections