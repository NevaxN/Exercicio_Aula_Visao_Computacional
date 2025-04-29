import cv2
import numpy as np

imagem = cv2.imread("./imagem2.jpg", cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(imagem, None)

imagem_sift = cv2.drawKeypoints(imagem, keypoints, None)

cv2.imshow('SIFT', imagem_sift)
cv2.waitKey(0)
cv2.destroyAllWindows()
