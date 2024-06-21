from joblib import load
import os
import numpy as np
from skimage import io
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte

# Carregar o modelo treinado
MODEL = load('models/svm_model_6_classes.joblib')  # Carregar o novo modelo

INPUT_DIR = 'image_nucleus'

# Função para calcular os descritores de Haralick
def calculate_haralick_features(image):
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
    return feature_vector

# Função para pré-processar a imagem
def preprocess_image(image_path):
    # Carregar a imagem em tons de cinza
    image = io.imread(image_path, as_gray=True)
    
    # Calcular as características de Haralick
    features = calculate_haralick_features(image)
    
    return features

# Função para processar todas as imagens em subpastas
def process_images_in_directory():
    predictions = []
    
    # Percorrer todas as subpastas
    for subdir, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png')):  # Ajuste conforme necessário
                image_path = os.path.join(subdir, file)
                
                # Pré-processar a imagem
                image_features = preprocess_image(image_path)
                
                # Fazer a previsão usando o novo modelo SVM
                prediction = MODEL.predict([image_features])
                
                # Armazenar o resultado
                predictions.append((image_path, prediction[0]))
    
    return predictions

# Chamar a função para processar as imagens e mostrar os resultados
if __name__ == "__main__":
    results = process_images_in_directory()
    
    # Imprimir os resultados
    print("\nResultados da Classificação de Imagens:\n")
    for image_path, prediction in results:
        print(f'Imagem: {image_path} - Previsão: Classe {prediction}')