import io
import tkinter as tk
import numpy as np
from skimage import io
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
import matplotlib.pyplot as plt


def calcular_descritores_haralick(caminho_arquivo):
    image = io.imread(caminho_arquivo, as_gray=True)
    distances = [1]  # Pode ser ajustado conforme necessário
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0, 45, 90, 135 graus
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

    # Converter a imagem para uint8
    image_uint8 = img_as_ubyte(image)
    
    glcm = graycomatrix(image_uint8, distances=distances, angles=angles, symmetric=True, normed=True)
    feature_vector = []
    for prop in properties:
        for angle in range(len(angles)):
            feature = graycoprops(glcm, prop)[0, angle]
            feature_vector.append(feature)
    num_properties = len(properties)
    num_angles = len(angles)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for i, prop in enumerate(properties):
        ax = axes.flat[i]
        bar_width = 0.15
        indices = np.arange(num_angles)
        colors = ['b', 'g', 'r', 'c']
        
        for j, angle in enumerate(angles):
            ax.bar(indices + j * bar_width, feature_vector[i::num_properties][j::num_angles], bar_width, label=f'{int(np.degrees(angle))}°', color=colors[j])
        
        ax.set_title(prop.capitalize())
        ax.set_xlabel('Ângulos')
        ax.set_ylabel('Valor')
        ax.set_xticks(indices + bar_width)
        ax.set_xticklabels([f'{int(np.degrees(angle))}°' for angle in angles])
        ax.legend(title='Ângulos')
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle('Descritores de Haralick', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def calcular_descritores_haralik_matriz_coocorencia(glcm):
    # Contraste
    contraste = graycoprops(glcm, 'contrast')[0, 0]
    
    # Homogeneidade
    homogeneidade = graycoprops(glcm, 'homogeneity')[0, 0]
    
    # Entropia
    with np.errstate(divide='ignore', invalid='ignore'):
        glcm_normalized = glcm / np.sum(glcm)
        entropy = -np.sum(glcm_normalized * np.log2(glcm_normalized + (glcm_normalized == 0)))
    
    return contraste, homogeneidade, entropy