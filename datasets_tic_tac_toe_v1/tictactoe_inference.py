from ultralytics import YOLO
import cv2
import pyrealsense2 as rs
import numpy as np
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_path = "best_tictactoe_v1.pt"
class_names = ['None', 'o', 'table', 'x']
target_class = 'table'

class ElevatorPanelDetector:
    TIMEOUT_DURATION = 20
    offset_x = 40
    offset_y = -5
    offset_z = 140
    pressed = False

    def __init__(self, model_path, class_names, target_class):
        self.model = YOLO("yolo11n-obb.pt")  # load an official model
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
        print("initialized")

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
                mask = np.zeros_like(color_image)
                results = self.model(source=color_image, conf=0.75, verbose=False, show=True)
                
                if results is not None:
                    print("Results:", results)
                    for result in results:
                        if hasattr(result, 'obb') and result.obb is not None:
                            obb = result.obb
                            print(obb)
                            if hasattr(obb, 'cls') and hasattr(obb, 'conf') and hasattr(obb, 'xyxy'):
                                class_ids = obb.cls.int().tolist()
                                confidences = obb.conf.tolist()
                                coordinates = obb.xyxy.int().tolist()
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
                        else:
                            print("No OBB")

                if mask is not None:
                    cv2.imshow('Mask', mask)
                else:
                    print("No mask")

                # Display the image with bounding boxes
                cv2.imshow('Inference', color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"Error: {e}")
            
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

def main():
    detector = ElevatorPanelDetector(model_path, class_names, target_class)
    detector.process_frames()

if __name__ == "__main__":
    main()