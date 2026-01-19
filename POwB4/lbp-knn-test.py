import cv2
import numpy as np
import os
import joblib
from skimage.feature import local_binary_pattern

def get_lbp_hist(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: 
        return None
    # Parametry muszą być IDENTYCZNE jak w skrypcie train.py
    radius, n_points = 3, 24
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

# 1. Wczytanie zapisanego modelu (wiedzy o 5 osobach)
if not os.path.exists('model_odciskow.pkl'):
    print("BŁĄD: Nie znaleziono pliku modelu! Uruchom najpierw skrypt trenujący.")
else:
    data = joblib.load('model_odciskow.pkl')
    knn = data['model']

    test_dir = 'test/'
    print(f"--- ROZPOZNAWANIE TOŻSAMOŚCI (Folder: {test_dir}) ---")

    # Sprawdzenie zawartości folderu
    pliki_testowe = [f for f in os.listdir(test_dir) if f.lower().endswith(".bmp")]

    if len(pliki_testowe) == 0:
        print("BŁĄD: Nie znaleziono plików .bmp w folderze /test")
    else:
        for filename in pliki_testowe:
            path = os.path.join(test_dir, filename)
            hist = get_lbp_hist(path)
            
            if hist is not None:
                # Klasyfikacja - k-NN szuka najbardziej podobnego histogramu
                prediction = knn.predict([hist])[0]
                
                # Wyciągamy ID z nazwy pliku testowego, żeby sprawdzić czy algorytm ma rację
                # Zakładamy, że plik testowy też nazywa się np. "1__M_Left..."
                true_id = filename.split("__")[0]
                
                wynik_tekst = "PRAWIDŁOWE" if prediction == true_id else "BŁĘDNE"
                
                print("-" * 30)
                print(f"Badany plik: {filename}")
                print(f"Rozpoznana osoba (ID): {prediction}")
                print(f"Faktyczna osoba  (ID): {true_id}")
                print(f"Werdykt: {wynik_tekst}")
                print("-" * 30)