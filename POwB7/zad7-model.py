import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split
import segmentation_models_pytorch as smp

# ==========================================
# 1. KONFIGURACJA
# ==========================================
IMG_DIR = "retinal_images/original_images/"
MASK_DIR = "masked_images/manual_images/"
OUTPUT_PREVIEW_DIR = "progress_images"
MODEL_SAVE_PATH = "best_retina_unet.pth"

IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 25
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tworzenie folderu na podgląd, jeśli nie istnieje
if not os.path.exists(OUTPUT_PREVIEW_DIR):
    os.makedirs(OUTPUT_PREVIEW_DIR)

# ==========================================
# 2. INTELIGENTNY DATASET (Obsługa JPG/PNG)
# ==========================================
class RetinaDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = os.path.abspath(img_dir)
        self.mask_dir = os.path.abspath(mask_dir)
        self.transform = transform
        
        # Parowanie plików po rdzeniu nazwy (ignorujemy rozszerzenie)
        img_files = {os.path.splitext(f)[0]: f for f in os.listdir(self.img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))}
        mask_files = {os.path.splitext(f)[0]: f for f in os.listdir(self.mask_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))}
        
        self.matched_names = sorted(list(set(img_files.keys()) & set(mask_files.keys())))
        self.img_map = img_files
        self.mask_map = mask_files
        
        if len(self.matched_names) == 0:
            raise RuntimeError(f"Błąd: Nie sparowano plików! Sprawdź foldery.")
        print(f"Dataset: Sparowano {len(self.matched_names)} obrazów.")

    def __len__(self):
        return len(self.matched_names)

    def __getitem__(self, idx):
        base_name = self.matched_names[idx]
        img_path = os.path.join(self.img_dir, self.img_map[base_name])
        mask_path = os.path.join(self.mask_dir, self.mask_map[base_name])
        
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (mask > 127).astype(np.float32) # Binaryzacja

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.unsqueeze(0)

# ==========================================
# 3. FUNKCJE POMOCNICZE (Wizualizacja i Metryki)
# ==========================================
def save_preview(model, dataset, epoch, device):
    model.eval()
    with torch.no_grad():
        image, mask = dataset[0] # Bierzemy pierwszy obraz z walidacji
        input_tensor = image.unsqueeze(0).to(device)
        output = torch.sigmoid(model(input_tensor))
        pred = (output > 0.5).float().cpu().squeeze().numpy()
        
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.title("Oryginał"); plt.imshow(img_np)
        plt.subplot(1, 3, 2); plt.title("Maska Ręczna"); plt.imshow(mask.squeeze(), cmap='gray')
        plt.subplot(1, 3, 3); plt.title(f"AI Epoka {epoch+1}"); plt.imshow(pred, cmap='gray')
        plt.savefig(f"{OUTPUT_PREVIEW_DIR}/epoch_{epoch+1}.png")
        plt.close()

def calculate_metrics(tp, fp, fn, tn):
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    return dice, iou

# ==========================================
# 4. PRZYGOTOWANIE TRENINGU
# ==========================================
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

dataset = RetinaDataset(IMG_DIR, MASK_DIR, transform=train_transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
val_set.dataset.transform = val_transform

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=1).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = smp.losses.DiceLoss(mode='binary')

# ==========================================
# 5. PĘTLA TRENINGOWA
# ==========================================
print(f"Start treningu na: {DEVICE}")
best_dice = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Walidacja i metryki
    model.eval()
    dices, ious = [], []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            preds = (torch.sigmoid(model(images)) > 0.5).float()
            tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), masks.long(), mode='binary')
            d, i = calculate_metrics(tp.sum(), fp.sum(), fn.sum(), tn.sum())
            dices.append(d.item()); ious.append(i.item())

    avg_dice = np.mean(dices)
    print(f"Epoka [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f} | Dice: {avg_dice:.4f} | IoU: {np.mean(ious):.4f}")
    
    save_preview(model, val_set, epoch, DEVICE) # Zapis podglądu PNG

    if avg_dice > best_dice:
        best_dice = avg_dice
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(" -> Model zapisany!")

print(f"Koniec. Najlepszy Dice: {best_dice:.4f}")