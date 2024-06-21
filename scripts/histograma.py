import tkinter as tk
import cv2
import numpy as np
import matplotlib.pyplot as plt

def gerar_histograma_hsv(caminho_arquivo):
    # Carrega a imagem
    imagem = cv2.imread(caminho_arquivo)
    if imagem is None:
        print("Erro ao carregar a imagem.")
        return
    
    # Converte a imagem de BGR para HSV
    imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    # Calcula os histogramas para cada canal
    hist_h = cv2.calcHist([imagem_hsv], [0], None, [256], [0, 256])
    hist_s = cv2.calcHist([imagem_hsv], [1], None, [256], [0, 256])
    hist_v = cv2.calcHist([imagem_hsv], [2], None, [256], [0, 256])

    # Plota os histogramas
    plt.figure(figsize=(10, 7))

    plt.subplot(3, 1, 1)
    plt.plot(hist_h, color='r')
    plt.title('Histograma do canal Hue')
    plt.xlabel('Intensidade')
    plt.ylabel('Contagem')

    plt.subplot(3, 1, 2)
    plt.plot(hist_s, color='g')
    plt.title('Histograma do canal Saturation')
    plt.xlabel('Intensidade')
    plt.ylabel('Contagem')

    plt.subplot(3, 1, 3)
    plt.plot(hist_v, color='b')
    plt.title('Histograma do canal Value')
    plt.xlabel('Intensidade')
    plt.ylabel('Contagem')

    plt.tight_layout()
    plt.show()

def gerar_histograma_tons_cinza(caminho_arquivo):
    # Carrega a imagem
    imagem = cv2.imread(caminho_arquivo, cv2.IMREAD_GRAYSCALE)
    
    # Calcula o histograma para o canal de intensidade
    hist = cv2.calcHist([imagem], [0], None, [256], [0, 256])

    # Plota o histograma
    plt.figure(figsize=(10, 7))
    plt.plot(hist, color='k')
    plt.title('Histograma da Imagem em Tons de Cinza')
    plt.xlabel('Intensidade')
    plt.ylabel('Contagem')
    plt.grid(True)
    plt.show()

def gerar_histograma_16tons(caminho_arquivo):
    imagem_gray = cv2.imread(caminho_arquivo, cv2.IMREAD_GRAYSCALE)

    # Reduzir a imagem para 16 tons de cinza
    imagem_gray = cv2.convertScaleAbs(imagem_gray, alpha=(15/255))

    # Calcular o histograma
    histograma = cv2.calcHist([imagem_gray], [0], None, [16], [0, 16])

    # Normalizar o histograma
    histograma /= histograma.sum()

    # Plotar o histograma
    plt.figure()
    plt.title("Histograma de Tons de Cinza")
    plt.xlabel("Intensidade")
    plt.ylabel("Quantidade de Pixels")
    plt.plot(histograma)
    plt.xlim([0, 16])

    # Exibir a imagem em 16 tons de cinza
    plt.figure()
    plt.title("Imagem em 16 Tons de Cinza")
    plt.imshow(imagem_gray, cmap='gray')
    plt.show()

def gerar_histograma_hiv(caminho_arquivo):
    img = cv2.imread(caminho_arquivo)

    # Converta a imagem para o espaço de cores HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Quantize o canal H para 16 valores e o canal V para 8 valores
    hsv[:,:,0] = (hsv[:,:,0]/180*16).astype(np.uint8)
    hsv[:,:,2] = (hsv[:,:,2]/255*8).astype(np.uint8)

    # Calcule o histograma 2D
    hist = cv2.calcHist([hsv], [0, 2], None, [16, 8], [0, 16, 0, 8])

    # Gere o gráfico do histograma
        # Gere o gráfico do histograma
    plt.imshow(hist, interpolation = 'nearest')
    plt.title('Histograma 2D')
    plt.xlabel('Canal H (Matiz)')
    plt.ylabel('Canal V (Valor)')
    plt.colorbar()
    plt.show()