# Codelab 1 — MTCNN en imágenes (versión integrada)
# Requiere: opencv-python, matplotlib, mtcnn, numpy

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from time import time

# --- 1. Preparación del entorno ---
# (asegúrate de tener instaladas las librerías con pip install opencv-python mtcnn matplotlib numpy)

# --- 2. Cargar imagen y detectar ---
img = cv2.imread('grupo.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

detector = MTCNN()
t0 = time()
res = detector.detect_faces(img_rgb)
t1 = time()
print(f"Detected: {len(res)} rostro(s) • tiempo: {(t1 - t0)*1000:.1f} ms")

# --- 3. Dibujar cajas y landmarks ---
for r in res:
    x, y, w, h = r['box']
    cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0,255,0), 2)
    for k, p in r['keypoints'].items():
        cv2.circle(img_rgb, p, 2, (255,0,0), -1)

plt.imshow(img_rgb)
plt.axis('off')
plt.show()

# --- 4. Umbrales y NMS ---
# Filtrar por confianza mínima
min_conf = 0.9
res_filtrados = [r for r in res if r['confidence'] >= min_conf]
print(f"Con umbral {min_conf}: {len(res_filtrados)} rostros")

# --- 5. IoU simple (evaluación rápida) ---
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interArea = max(0, xB-xA) * max(0, yB-yA)
    boxAArea = boxA[2]*boxA[3]
    boxBArea = boxB[2]*boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)

if len(res_filtrados) >= 2:
    iou_val = iou(res_filtrados[0]['box'], res_filtrados[1]['box'])
    print("IoU entre los dos primeros rostros:", round(iou_val, 3))

# --- 6. Tareas finales ---
for i, r in enumerate(res_filtrados):
    print(f"Rostro {i+1}: confianza={r['confidence']:.3f}, box={r['box']}")
