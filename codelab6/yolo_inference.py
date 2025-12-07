import cv2
from ultralytics import YOLO

model_yolo = YOLO("yolov8n.pt")
results = model_yolo("image.png")

img = results[0].plot()   # Devuelve un array con la imagen anotada
cv2.imshow("YOLOv8", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
