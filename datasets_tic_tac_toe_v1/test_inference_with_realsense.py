from ultralytics import YOLO
import cv2
import pyrealsense2 as rs
import numpy as np
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_path = "/home/smr/ultralytics/datasets_tic_tac_toe_v1/best_tictactoe_v1.pt"
class_names = ['null', 'o', 'table', 'x']
target_class = 'table'

class ElevatorPanelDetector:
    TIMEOUT_DURATION = 20
    offset_x = 40
    offset_y = -5
    offset_z = 140
    pressed = False

    def __init__(self, model_path, class_names, target_class):
        self.model = YOLO(model_path)
        self.class_names = class_names
        self.target_class = target_class
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)
        self.pipeline.start(self.config)
        self.depth_scale = self.pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
        self.stop_event = threading.Event()

    def process_frames(self):
        try:
            while not self.stop_event.is_set():
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                results = self.model.predict(source=color_image, conf=0.75, verbose=True, show=False)
                
                for result in results:
                    boxes = result.boxes
                    class_ids = boxes.cls.int().tolist()
                    confidences = boxes.conf.tolist()
                    coordinates = boxes.xyxy.int().tolist()
                    for class_id, confidence, (x1, y1, x2, y2) in zip(class_ids, confidences, coordinates):
                        class_name = self.class_names[class_id]
                        print(f"Class: {class_name}, Confidence: {confidence}, Coordinates: {x1, y1, x2, y2}")
                        if class_name == self.target_class:
                            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            roi = color_image[y1:y2, x1:x2]
                            mask = np.zeros_like(color_image)
                            mask[y1:y2, x1:x2] = roi

                            cv2.putText(color_image, f"{class_name}: {confidence:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if mask is not None:
                    cv2.imshow('Mask', mask)

                
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

def main():
    detector = ElevatorPanelDetector(model_path, class_names, target_class)
    detector.process_frames()

if __name__ == "__main__":
    main()