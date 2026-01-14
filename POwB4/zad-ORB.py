import cv2
import matplotlib.pyplot as plt

def compare_fingerprints(img1_path, img2_path):
    # 1. Wczytanie obrazów w skali szarości
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Błąd: Nie można wczytać plików obrazów.")
        return

    # 2. Inicjalizacja detektora ORB
    orb = cv2.ORB_create(nfeatures=1000)

    # 3. Wykrywanie punktów kluczowych i obliczanie deskryptorów
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 4. Dopasowanie punktów za pomocą Brute-Force Matcher (używając normy Hamminga)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # 5. Sortowanie dopasowań według dystansu (najlepsze dopasowania na początku)
    matches = sorted(matches, key=lambda x: x.distance)

    # 6. Wizualizacja wyników (wyświetlamy 30 najlepszych dopasowań)
    result_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, 
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(15, 7))
    plt.imshow(result_img)
    plt.title(f"Liczba dopasowań: {len(matches)}")
    plt.show()

if __name__ == "__main__":
    compare_fingerprints("org/1__M_Left_index_finger_CR.bmp", "alt/1__M_Left_index_finger_CR.bmp")