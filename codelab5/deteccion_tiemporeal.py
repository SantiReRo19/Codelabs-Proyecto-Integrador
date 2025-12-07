# deteccion_tiempo_real.py
import os
import cv2
import json
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Crear carpeta de resultados si no existe
os.makedirs("resultados", exist_ok=True)

# Cargar modelo YOLO nano
model = YOLO("yolov8n.pt")

# Inicializar webcam
cap = cv2.VideoCapture(0)
frame_count = 0
conteo_personas = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()

    # Exportar detecciones a JSON
    detecciones = []
    personas_en_frame = 0
    for r in results[0].boxes:
        clase = model.names[int(r.cls)]
        if clase == "person":
            personas_en_frame += 1
        obj = {
            "clase": clase,
            "score": float(r.conf),
            "bbox": r.xyxy.tolist()[0]
        }
        detecciones.append(obj)

    # Guardar detecciones por frame en carpeta
    with open(f"resultados/frame_{frame_count}.json", "w") as f:
        json.dump(detecciones, f, indent=4)

    conteo_personas.append(personas_en_frame)
    frame_count += 1

    cv2.imshow("Detección YOLOv8", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# --- Reto: gráfico de personas detectadas por frame ---
plt.plot(conteo_personas, marker="o")
plt.title("Personas detectadas por frame")
plt.xlabel("Frame")
plt.ylabel("# Personas")
plt.grid(True)
plt.savefig("resultados/grafico_personas.png")
plt.show()

print("✅ Gráfico generado en carpeta 'resultados/grafico_personas.png'")
