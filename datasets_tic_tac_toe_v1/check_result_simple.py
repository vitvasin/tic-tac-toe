from ultralytics import YOLO
import cv2
import os

# Load the custom-trained model
model = YOLO("best_tictactoe_v1.pt")  # replace with the path to your custom-trained model

# Define the target class to filter
model_path = "best_tictactoe_v1.pt"
class_names = ['null', 'o', 'table', 'x']
target_class = 'table'

# Directory containing images
image_dir = '/home/smr/Desktop/test_data'  # replace with your directory path

# Iterate through all files in the directory
for filename in os.listdir(image_dir):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        results = model.predict(source=img, conf=0.5, show=True)

        # # Iterate through the results and filter by target class
        # for result in results:
        #     boxes = result.boxes  # Access the detected boxes
        #     class_ids = boxes.cls.int().tolist()  # Get the class IDs as a list of integers
        #     confidences = boxes.conf.tolist()  # Get the confidences as a list of floats
        #     coordinates = boxes.xyxy.int().tolist()  # Get bounding box coordinates as a list of integers

        #     for class_id, confidence, (x1, y1, x2, y2) in zip(class_ids, confidences, coordinates):
        #         class_name = class_names[class_id]  # Map ID to class name
        #         if class_name == target_class:
        #             # Draw the bounding box on the image
        #             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
        #             cv2.putText(img, f"{class_name}: {confidence:.2f}", (x1, y1 - 10),
        #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the image
        cv2.imshow('Image', img)
        cv2.waitKey(0)

cv2.destroyAllWindows()

