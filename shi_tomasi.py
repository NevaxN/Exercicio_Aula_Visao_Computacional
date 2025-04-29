import cv2
import numpy as np

def carregar_imagem(nome_imagem):
    return cv2.imread(nome_imagem, cv2.IMREAD_GRAYSCALE)

def aplicar_algoritmo(imagem, mc, ql, md):
    cantos = cv2.goodFeaturesToTrack(
        imagem, 
        maxCorners=100, 
        qualityLevel=0.04, 
        minDistance=10
    )

    cantos = np.int64(cantos)

    for canto in cantos:
        x, y = canto.ravel()
        cv2.circle(imagem, (x, y), 3, 255, -1)

    return imagem

def display_imagem(imagem):
    cv2.imshow('Shi-Tomasi', imagem)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


imagem1 = aplicar_algoritmo(carregar_imagem("./imagem1.jpg"), 100, 0.04, 10)
display_imagem(imagem=imagem1)

imagem2 = aplicar_algoritmo(carregar_imagem("./imagem2.jpg"), 500, 1, 50)
display_imagem(imagem=imagem2)

imagem3 = aplicar_algoritmo(carregar_imagem("./imagem3.jpg"), 200, 0.07, 25)
display_imagem(imagem=imagem3)