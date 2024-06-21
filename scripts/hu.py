import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, color, io
from skimage.measure import moments_hu
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para calcular momentos invariantes de Hu
def calculate_hu_moments(image):
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments)
    # Transformando os momentos em uma escala logarítmica
    for i in range(0, 7):
        hu_moments[i] = -1 * np.sign(hu_moments[i]) * np.log10(np.abs(hu_moments[i]))
    return hu_moments.flatten()

def hu_cinza_hsv(caminho_arquivo):
    # Carregue a imagem
    image = cv2.imread(caminho_arquivo)

    # Converta a imagem para tons de cinza
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calcule os momentos invariantes de Hu para a imagem em tons de cinza
    hu_moments_gray = calculate_hu_moments(gray_image)

    print("Momentos Invariantes de Hu - Imagem em Tons de Cinza:")
    print(hu_moments_gray)

    # Calcule os momentos invariantes de Hu para cada canal de cor (R, G, B)
    channels = cv2.split(image)
    hu_moments_channels = []
    for channel in channels:
        hu_moments_channel = calculate_hu_moments(channel)
        hu_moments_channels.append(hu_moments_channel)
        print(f"Momentos Invariantes de Hu - Canal {len(hu_moments_channels)}:")
        print(hu_moments_channel)

    # Plot para imagem em tons de cinza e canais de cor em uma tela
    labels = ['Hu Moment 1', 'Hu Moment 2', 'Hu Moment 3', 'Hu Moment 4', 'Hu Moment 5', 'Hu Moment 6', 'Hu Moment 7']
    x = np.arange(len(labels))

    plt.figure(figsize=(15, 10))

    # Plot para imagem em tons de cinza
    plt.subplot(2, 2, 1)
    plt.bar(x, hu_moments_gray, color='gray')
    plt.xlabel('Hu Moments')
    plt.ylabel('Valor')
    plt.title('Momentos Invariantes de Hu - Imagem em Tons de Cinza')
    plt.xticks(x, labels)

    # Plot para canal Vermelho
    plt.subplot(2, 2, 2)
    plt.bar(x, hu_moments_channels[0], color='red')
    plt.xlabel('Hu Moments')
    plt.ylabel('Valor')
    plt.title('Momentos Invariantes de Hu - Canal Vermelho')
    plt.xticks(x, labels)

    # Plot para canal Verde
    plt.subplot(2, 2, 3)
    plt.bar(x, hu_moments_channels[1], color='green')
    plt.xlabel('Hu Moments')
    plt.ylabel('Valor')
    plt.title('Momentos Invariantes de Hu - Canal Verde')
    plt.xticks(x, labels)

    # Plot para canal Azul
    plt.subplot(2, 2, 4)
    plt.bar(x, hu_moments_channels[2], color='blue')
    plt.xlabel('Hu Moments')
    plt.ylabel('Valor')
    plt.title('Momentos Invariantes de Hu - Canal Azul')
    plt.xticks(x, labels)

    plt.tight_layout()
    plt.show()


def gerar_invariantes_hu_256(caminho_arquivo):
    def calcular_momentos_hu(imagem):
        # Calcular os momentos invariantes de Hu
        momentos = moments_hu(imagem)
        return momentos

    # Carregar a imagem
    img_original = io.imread(caminho_arquivo)
    
    # Converter a imagem para escala de cinza com 256 tons
    img_gray = color.rgb2gray(img_original)
    img_gray = exposure.rescale_intensity(img_gray, out_range=(0, 255)).astype(np.uint8)
    
    # Converter a imagem para o modelo HSV
    img_hsv = color.rgb2hsv(img_original)
    
    # Extrair os canais HSV
    h_channel = img_hsv[:, :, 0]
    s_channel = img_hsv[:, :, 1]
    v_channel = img_hsv[:, :, 2]
    
    # Calcular os momentos invariantes de Hu para a imagem em 256 tons de cinza
    momentos_hu_gray = calcular_momentos_hu(img_gray)
    
    # Calcular os momentos invariantes de Hu para cada canal do modelo HSV
    momentos_hu_h = calcular_momentos_hu(h_channel)
    momentos_hu_s = calcular_momentos_hu(s_channel)
    momentos_hu_v = calcular_momentos_hu(v_channel)
    
    # Função para exibir os momentos calculados
    def exibir_momentos(momentos, descricao):
        print(f"\nMomentos invariantes de Hu para {descricao}:")
        for i, momento in enumerate(momentos):
            print(f"Momento {i+1}: {momento:.5e}")

    # Exibir os momentos calculados
    exibir_momentos(momentos_hu_gray, "a imagem em 256 tons de cinza")
    exibir_momentos(momentos_hu_h, "o canal H do modelo HSV")
    exibir_momentos(momentos_hu_s, "o canal S do modelo HSV")
    exibir_momentos(momentos_hu_v, "o canal V do modelo HSV")
    
    # Plotar os momentos calculados
    def plotar_momentos(momentos_gray, momentos_h, momentos_s, momentos_v):
        momentos = [momentos_gray, momentos_h, momentos_s, momentos_v]
        descricoes = ["Gray Image", "H Channel", "S Channel", "V Channel"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()  # Flatten the 2D array of axes

        for i, (momento, descricao) in enumerate(zip(momentos, descricoes)):
            axes[i].bar(range(1, 8), momento)
            axes[i].set_title(f'Momentos Hu - {descricao}')
            axes[i].set_xlabel('Momento')
            axes[i].set_ylabel('Valor')
            for j, val in enumerate(momento):
                axes[i].text(j + 1, val, f'{val:.2e}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    plotar_momentos(momentos_hu_gray, momentos_hu_h, momentos_hu_s, momentos_hu_v)