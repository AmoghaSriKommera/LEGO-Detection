from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Train on your LEGO dataset
model.train(
    data="lego_dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    project="lego_yolo",
    name="exp1"
)
