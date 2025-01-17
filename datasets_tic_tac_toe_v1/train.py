from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-obb.pt")  # load a pretrained model (recommended for training)

# Train the model
#results = model.train(data="/home/smr/ultralytics/datasets_tic_tac_toe_v1/data.yaml", epochs=300, imgsz=640, batch=4, device=0, half=True)

results = model.train(data="/home/smr/ultralytics/datasets_tic_tac_toe_v1/data.yaml", epochs=100, imgsz=640, batch=0.9, device=0, half=True)
