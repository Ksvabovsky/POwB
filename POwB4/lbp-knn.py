import cv2
import numpy as np
import os
import joblib
from skimage.feature import local_binary_pattern
from sklearn.neighbors import KNeighborsClassifier

def get_lbp_hist(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    radius, n_points = 3, 24
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

X_train, y_train = [], []
train_dir = 'org/'

for filename in os.listdir(train_dir):
    # pliki
    if filename.lower().endswith((".bmp", ".jpg", ".jpeg", ".png")):
        person_id = filename.split("__")[0]
        path = os.path.join(train_dir, filename)
        
        hist = get_lbp_hist(path)
        if hist is not None:
            X_train.append(hist)
            y_train.append(person_id)
            print(f"Wczytano: {filename}")

# Trening modelu
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train)

# Zapisanie modelu i danych do pliku
joblib.dump({'model': knn, 'features': X_train, 'labels': y_train}, 'model_odciskow.pkl')
print(f"Sukces! Model został zapisany jako 'model_odciskow.pkl'.")