import cv2
import numpy as np

def carregar_imagem(nome_imagem):
    return cv2.imread(nome_imagem, cv2.IMREAD_GRAYSCALE)

def aplicar_algoritmo(imagem, bs, ks, k):
    img_copy = imagem.copy()
    harris = cv2.cornerHarris(img_copy, blockSize=bs, ksize=ks, k=k)
    harris = cv2.dilate(harris, None)
    img_copy[harris > 0.01 * harris.max()] = 255
    return img_copy

def display_imagem(imagem):
    cv2.imshow('Harris Corner', imagem)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

imagem1 = aplicar_algoritmo(carregar_imagem("./imagem1.jpg"), 2, 3, 0.04)
display_imagem(imagem=imagem1)

imagem2 = aplicar_algoritmo(carregar_imagem("./imagem2.jpg"), 3, 5, 0.03)
display_imagem(imagem=imagem2)

imagem3 = aplicar_algoritmo(carregar_imagem("./imagem3.jpg"), 5, 7, 0.05)
display_imagem(imagem=imagem3)
