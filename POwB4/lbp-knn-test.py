import cv2
import numpy as np
import os
import joblib
from skimage.feature import local_binary_pattern

# testowy odcisk do histogramu
def get_lbp_hist(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: 
        return None
  
    radius, n_points = 3, 24
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

# wczytanie modelu z joblib
if not os.path.exists('model_odciskow.pkl'):
    print("BLAD: nie znaleziono pliku modelu, uruchom najpierw skrypt trenujacy.")
else:
    data = joblib.load('model_odciskow.pkl')
    knn = data['model']

    test_dir = 'test/'
    print(f"--- ROZPOZNAWANIE TOZSAMOSCI (Folder: {test_dir}) ---")

    pliki_testowe = [f for f in os.listdir(test_dir) if f.lower().endswith(".bmp")]

    if len(pliki_testowe) == 0:
        print("BLAD: Nie znaleziono plikow .bmp w folderze /test")
    else:
        for filename in pliki_testowe:
            path = os.path.join(test_dir, filename)
            hist = get_lbp_hist(path)
            
            if hist is not None:
                # klasyfikacja przez knn
                prediction = knn.predict([hist])[0]
                
                true_id = filename.split("__")[0]
                
                wynik_tekst = "PRAWIDLOWE" if prediction == true_id else "BLEDNE"
                
                print("-" * 30)
                print(f"badany plik: {filename}")
                print(f"rozpoznana osoba (ID): {prediction}")
                print(f"faktyczna osoba  (ID): {true_id}")
                print(f"rezultat: {wynik_tekst}")
                print("-" * 30)