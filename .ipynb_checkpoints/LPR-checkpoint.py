import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract
import skimage
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

class LPR:
    def __init__(self, min_w=80, max_w=360, min_h=25, max_h=130, ratio=2.769230769230769):
        self.min_w = min_w
        self.max_w = max_w
        self.min_h = min_h
        self.max_h = max_h
        self.ratio = ratio

    def grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def apply_threshold(self, img): # Con otsu aplica un threshold especifico para cada imaggen
        _, thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) 
        return thresh_img

    def apply_adaptive_threshold(self, img):
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 13)

    def find_contours(self, img):
        return cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    def filter_candidates(self, img, contours):
        candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter criteria based on size
            if self.min_w <= w <= self.max_w and self.min_h <= h <= self.max_h:
                # Extract the region of interest
                roi = img[y:y+h, x:x+w]
                
                # Apply OCR to the region of interest
                text = pytesseract.image_to_string(roi, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                
                # If OCR detects alphanumeric characters, consider it a candidate
                if any(char.isalnum() for char in text):
                    candidates.append(cnt)
        return candidates

    def get_lowest_candidate(self, candidates):
    square_candidates = []
    ys = []
    for cnt in candidates:
        x, y, w, h = cv2.boundingRect(cnt)
        # Calcula la relación de aspecto
        aspect_ratio = float(w) / h
        # Filtra por contornos cuadrados con una tolerancia en la relación de aspecto
        if 0.9 <= aspect_ratio <= 1.1:
            square_candidates.append(cnt)
            ys.append(y)
    if not square_candidates:
        return None
    # Retorna el contorno cuadrado más bajo
    return square_candidates[np.argmax(ys)]

    def crop_license_plate(self, img, license):
        x, y, w, h = cv2.boundingRect(license)
        return img[y:y+h, x:x+w]

    def clear_border(self, img):
        return skimage.segmentation.clear_border(img)

    def invert_image(self, img):
        return cv2.bitwise_not(img)

    def read_license(self, img, psm=7):
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        options += " --psm {}".format(psm)

        gray = self.grayscale(img)
        thresh = self.apply_threshold(gray)
        contours = self.find_contours(thresh)
        candidates = self.filter_candidates(contours)
        if candidates:
            license = candidates[0]
            if len(candidates) > 1:
                license = self.get_lowest_candidate(candidates)
            cropped = self.crop_license_plate(gray, license)
            thresh_cropped = self.apply_adaptive_threshold(cropped)
            clear_border = self.clear_border(thresh_cropped)
            final = self.invert_image(clear_border)
            txt = pytesseract.image_to_string(final, config=options)
            return txt
        else:
            return "No license plate found"
