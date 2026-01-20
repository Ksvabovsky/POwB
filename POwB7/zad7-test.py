import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import random

# ==========================================
# 1. KONFIGURACJA
# ==========================================
IMG_DIR = "retinal_images/original_images/"
MASK_DIR = "masked_images/manual_images/"
MODEL_PATH = "best_retina_unet.pth"
IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. FUNKCJE POMOCNICZE
# ==========================================
def calculate_metrics(pred, target):
    # pred i target to macierze numpy (0 lub 1)
    intersection = np.logical_and(pred, target).sum()
    total = pred.sum() + target.sum()
    union = np.logical_or(pred, target).sum()
    
    dice = (2. * intersection) / (total + 1e-7)
    iou = intersection / (union + 1e-7)
    return dice, iou

# Preprocessing (musi byc taki sam jak podczas treningu)
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ==========================================
# 3. ŁADOWANIE MODELU I DANYCH
# ==========================================
# Inicjalizacja architektury
model = smp.Unet(encoder_name="resnet18", in_channels=3, classes=1).to(DEVICE)

# Wczytanie wag
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Nie znaleziono pliku modelu: {MODEL_PATH}. Najpierw wytrenuj model!")

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Pobranie listy plikow (parowanie nazw)
img_files = {os.path.splitext(f)[0]: f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))}
mask_files = {os.path.splitext(f)[0]: f for f in os.listdir(MASK_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))}
matched_names = list(set(img_files.keys()) & set(mask_files.keys()))

# ==========================================
# 4. EKSTRAKCJA I WIZUALIZACJA
# ==========================================
# Wybór losowego zdjęcia
random_name = random.choice(matched_names)
print(f"Testowanie na obrazie: {random_name}")

img_path = os.path.join(IMG_DIR, img_files[random_name])
mask_path = os.path.join(MASK_DIR, mask_files[random_name])

# Wczytanie i przygotowanie oryginału
image_raw = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
input_tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(DEVICE)

# Wczytanie i przygotowanie maski (Ground Truth)
mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask_resized = cv2.resize(mask_raw, (IMG_SIZE, IMG_SIZE))
mask_gt = (mask_resized > 127).astype(np.float32)

# Predykcja modelu
with torch.no_grad():
    output = torch.sigmoid(model(input_tensor))
    pred_mask = (output > 0.5).float().cpu().squeeze().numpy()

# Obliczanie metryk
dice, iou = calculate_metrics(pred_mask, mask_gt)

# Wyświetlanie wyników
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Oryginał (Siatkówka)")
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Maska Manualna (Ground Truth)")
plt.imshow(mask_gt, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title(f"Ekstrakcja AI\nDice: {dice:.4f}, IoU: {iou:.4f}")
plt.imshow(pred_mask, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Wynik dla tej próby -> Dice: {dice:.4f}, IoU: {iou:.4f}")