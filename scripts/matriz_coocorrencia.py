import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix
from skimage import exposure, color, io
import haralick

def gerar_matriz_coocorrencia(caminho_arquivo):
    img_mat = io.imread(caminho_arquivo)

    # Carregar a imagem em escala de cinza
    img_mat = color.rgb2gray(img_mat)

    # Reduzir a imagem para 16 níveis de cinza
    img_mat = exposure.rescale_intensity(img_mat, out_range=(0, 15)).astype(np.uint8)

    # Definir os níveis de cinza (0-15 para 16 níveis)
    levels = 16

    # Definir as distâncias específicas
    distances = [1, 2, 4, 8, 16, 32]
    angles = [0]  # Considerando apenas o ângulo de 0 graus para simplificação

    # Inicializar a matriz para armazenar as matrizes de co-ocorrência
    glcm_matrices = {}
    descritores = []

    # Calcular a matriz de co-ocorrência para cada distância
    for d in distances:
        glcm = graycomatrix(img_mat, [d], angles, levels, symmetric=True, normed=True)
        glcm_matrices[d] = glcm[:, :, 0, 0]  # Extraindo a matriz para o ângulo 0
        
        # Calcular descritores
        contraste, homogeneidade, entropia = haralick.calcular_descritores_haralik_matriz_coocorencia(glcm)
        descritores.append((contraste, homogeneidade, entropia))
        print(glcm)
    
    # Visualizar as matrizes de co-ocorrência
    fig, axes = plt.subplots(2, len(distances) // 2, figsize=(13, 8), constrained_layout=True)
    axes = axes.ravel()  # Flatten the 2D array of axes

    for i, d in enumerate(distances):
        im = axes[i].imshow(glcm_matrices[d], cmap='gray', vmin=0, vmax=np.max(glcm_matrices[d]))
        contraste, homogeneidade, entropia = descritores[i]
        axes[i].set_title(f'Distance: {d}\nContraste: {contraste:.4f}\nHomogeneidade: {homogeneidade:.4f}\nEntropia: {entropia:.4f}')
        axes[i].set_xlabel('Gray Level')
        axes[i].set_ylabel('Gray Level')
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.show()