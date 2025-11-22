from ultralytics import YOLO

model = YOLO("models/best.pt")

results = model("models/101.webp")

results[0].show()  # Affiche les detections
