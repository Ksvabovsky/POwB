import cv2
import numpy as np
import os

def extract_vessels_simple(img_path):
    # 1. Wczytanie i kanał zielony (tego nie pomijamy, bo to podstawa)
    img = cv2.imread(img_path)
    if img is None: return None
    green = img[:, :, 1]
    
    # 2. Rozmycie (Median Blur) - usuwa drobny szum "sól i pieprz"
    # To znacznie prostsze niż filtry Gaussa czy CLAHE
    blurred = cv2.medianBlur(green, 5)
    
    # 3. Progowanie adaptacyjne (Adaptive Thresholding)
    # Zastępuje CLAHE, Top-Hat i zwykłą binaryzację jednym krokiem.
    # Oblicza próg dla małych regionów, co świetnie radzi sobie z nierównym oświetleniem.
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # 4. Lekkie czyszczenie (opcjonalne, ale warto zostawić)
    kernel = np.ones((2,2), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return clean

# --- LOGIKA ZAPISU I METRYK (taka sama jak wcześniej) ---

path_images = r'retinal_images/original_images'
path_manual = r'masked_images/manual_images'
path_output = r'masked_images/new_images'

if not os.path.exists(path_output): os.makedirs(path_output)

dice_list, iou_list = [], []
files = sorted([f for f in os.listdir(path_images) if f.lower().endswith(('.jpg', '.png', '.tif'))])

for filename in files:
    img_path = os.path.join(path_images, filename)
    gt_path = os.path.join(path_manual, filename)
    
    # Uruchomienie uproszczonego algorytmu
    predicted_mask = extract_vessels_simple(img_path)
    
    if predicted_mask is not None:
        cv2.imwrite(os.path.join(path_output, filename), predicted_mask)
        
        if os.path.exists(gt_path):
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                if predicted_mask.shape != gt.shape:
                    gt = cv2.resize(gt, (predicted_mask.shape[1], predicted_mask.shape[0]))
                
                # Obliczanie metryk
                pred_b, gt_b = predicted_mask > 0, gt > 0
                inter = np.logical_and(pred_b, gt_b).sum()
                union = np.logical_or(pred_b, gt_b).sum()
                dice = (2.0 * inter) / (pred_b.sum() + gt_b.sum()) if (pred_b.sum() + gt_b.sum()) > 0 else 0
                iou = inter / union if union > 0 else 0
                
                dice_list.append(dice)
                iou_list.append(iou)
                print(f"{filename:<20} | Dice: {dice:.4f} | IoU: {iou:.4f}")

if dice_list:
    print("-" * 50)
    print(f"ŚREDNI DICE: {np.mean(dice_list):.4f}")
    print(f"ŚREDNI IoU:  {np.mean(iou_list):.4f}")