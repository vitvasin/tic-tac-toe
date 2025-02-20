import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

model_path = "/home/stu/tic-tac-toe_test_yolo_20.2.2025/tic-tac-toe_test_yolo/datasets_tic_tac_toe_v1/best_tictactoe_v1.pt"
class_names = ['None', 'o', 'table', 'x']
class_names_x = ['table','null']

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
pipeline.start(config)

model = YOLO(model_path)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        results = model.predict(source=color_image, conf=0.8, verbose=True, show=True)

        # for result in results:
        #     boxes = result.boxes
        #     if boxes is not None:
        #         for box in boxes:
        #             x1, y1, x2, y2 = box.xyxy.int().tolist()
        #             class_id = box.cls.int().item()
        #             class_name = class_names[class_id]
        #             confidence = box.conf.item()
        #             cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #             cv2.putText(color_image, f"{class_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        #cv2.imshow('RealSense', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()