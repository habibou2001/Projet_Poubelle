from ultralytics import YOLO

# Charger un modèle pré-entraîné
model = YOLO("yolov8n.pt")  # léger et rapide

# Entraînement
model.train(
    data="detection_poubelle.v1i.yolov8/data.yaml",   # chemin vers ton dataset
    epochs=50,
    imgsz=640,
    batch=8,
    name="trash_detector",
    pretrained=True
)
