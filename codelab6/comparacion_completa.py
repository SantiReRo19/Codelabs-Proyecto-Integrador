import torch
import time
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from ultralytics import YOLO
from PIL import Image

# --- 1. CONFIGURACIÓN ---
imagenes_a_probar = ["image.png", "image2.png", "image3.png"] # <--- CAMBIA ESTO SI TUS NOMBRES SON DISTINTOS

# --- 2. FUNCIÓN IOU ---
def calcular_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0

# --- 3. CARGA DE MODELOS (Se hace fuera del bucle para no afectar el tiempo de medición) ---
print("Cargando modelos... espera un momento...")

# SSD
weights = SSD300_VGG16_Weights.DEFAULT
model_ssd = ssd300_vgg16(weights=weights).eval()
preprocess_ssd = weights.transforms()

# YOLO
model_yolo = YOLO("yolov8n.pt")

print("\n" + "="*85)
print(f"{'IMAGEN':<15} | {'T. SSD (s)':<10} | {'T. YOLO (s)':<10} | {'OBJ SSD':<8} | {'OBJ YOLO':<8} | {'IoU (Top 1)':<10}")
print("="*85)

# --- 4. BUCLE DE PROCESAMIENTO ---
for img_path in imagenes_a_probar:
    try:
        # Preparar imagen
        original_img = Image.open(img_path).convert("RGB")
        
        # --- EJECUCIÓN SSD ---
        input_tensor = preprocess_ssd(original_img).unsqueeze(0)
        with torch.no_grad():
            t0 = time.time()
            out = model_ssd(input_tensor)[0]
            t1 = time.time()
        
        time_ssd = t1 - t0
        # Filtrar objetos SSD con confianza > 0.5 para contar solo detecciones reales
        indices_validos = [i for i, s in enumerate(out["scores"]) if s > 0.5]
        num_obj_ssd = len(indices_validos)
        
        # Obtener la mejor caja SSD para el IoU
        box_ssd = out["boxes"][0].numpy() if len(out["boxes"]) > 0 else None

        # --- EJECUCIÓN YOLO ---
        t0 = time.time()
        results = model_yolo(img_path, verbose=False)
        t1 = time.time()
        
        time_yolo = t1 - t0
        num_obj_yolo = len(results[0].boxes)
        
        # Obtener la mejor caja YOLO para el IoU
        box_yolo = results[0].boxes.xyxy[0].cpu().numpy() if len(results[0].boxes) > 0 else None

        # --- CALCULAR IOU ---
        if box_ssd is not None and box_yolo is not None:
            iou_val = calcular_iou(box_ssd, box_yolo)
        else:
            iou_val = 0.0

        # --- IMPRIMIR FILA DE LA TABLA ---
        print(f"{img_path:<15} | {time_ssd:<10.4f} | {time_yolo:<10.4f} | {num_obj_ssd:<8} | {num_obj_yolo:<8} | {iou_val:<10.4f}")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {img_path}")

print("="*85)