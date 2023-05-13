import pytesseract
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# list_imgs = ['training-strips/cartoon1.png'] #Take in input most likely
# transcript = []
# for fold in list_imgs:
#     img = cv2.imread(fold)
#     img = cv2.resize(img, (600,360))
#     transcript.extend(pytesseract.image_to_string(img))

# print(transcript)

img = cv2.imread("training-strips/cartoon1.png")
img = cv2.resize(img, (600, 360))
kernel = np.ones((1, 1), np.uint8)
img = cv2.dilate(img, kernel, iterations=1)
img = cv2.erode(img, kernel, iterations=1)

print(pytesseract.image_to_string(img))
cv2.imshow("Result", img)
