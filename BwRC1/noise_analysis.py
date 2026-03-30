import cv2
import os
import numpy as np

# Pliki konfiguracyjne BRISQUE
MODEL = "brisque_model_live.yml"
RANGE = "brisque_range_live.yml"
# Ścieżka do głównego folderu ze zdjęciami
BASE_DIR = "data/images"

def analyze_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    
    # 1. SZUM (Metoda Laplaciana)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_sigma = np.std(laplacian)
    noise_pct = (noise_sigma / 255.0) * 100
    
    # 2. BRISQUE (OpenCV Quality Module)
    try:
        brisque_eval = cv2.quality.QualityBRISQUE_create(MODEL, RANGE)
        score = brisque_eval.compute(img)[0]
    except Exception as e:
        score = -1.0  # Oznaczenie błędu
        
    return noise_pct, score

def run_full_analysis():
    if not os.path.exists(BASE_DIR):
        print(f"Błąd: Nie znaleziono folderu {BASE_DIR}")
        return

    print(f"{'Osoba':<20} | {'Plik':<12} | {'Szum [%]':<10} | {'BRISQUE'}")
    print("-" * 60)

    # Przeszukiwanie wszystkich podfolderów
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(root, file)
                
                # Pobieramy nazwę folderu osoby (ostatni człon ścieżki)
                person_name = os.path.basename(root)
                
                result = analyze_image(full_path)
                
                if result:
                    noise, brisque_val = result
                    print(f"{person_name[:20]:<20} | {file:<12} | {noise:>8.2f}% | {brisque_val:>8.2f}")

if __name__ == "__main__":
    run_full_analysis()