import cv2
import matplotlib.pyplot as plt

def compare_fingerprints(img1_path, img2_path):
    # wczytanie obrazow w skali szarosci
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Błąd: Nie można wczytać plików obrazów.")
        return

    # init ORB
    orb = cv2.ORB_create(nfeatures=1000)

    
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # dopasowanie Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # sortowanie dopasowan wedlug dystansu (najlepsze dopasowania na poczatku)
    matches = sorted(matches, key=lambda x: x.distance)

    # wizualizacja wynikow, wyswietlamy 30 najlepszych dopasowan
    result_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, 
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(15, 7))
    plt.imshow(result_img)
    plt.title(f"Liczba dopasowań: {len(matches)}")
    plt.show()

if __name__ == "__main__":
    compare_fingerprints("org/1__M_Left_index_finger_CR.bmp", "alt/1__M_Left_index_finger_CR.bmp")