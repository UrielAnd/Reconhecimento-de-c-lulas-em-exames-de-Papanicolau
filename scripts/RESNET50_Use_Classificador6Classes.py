import os
import numpy as np
from skimage import io, img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Diretório onde as novas imagens estão armazenadas
INPUT_DIR = 'image_nucleus'

# Carregar o modelo treinado com TensorFlow Keras (ResNet50) para 6 classes
MODEL = load_model('models/resnet50_best_model_6_classes.keras')

# Função para calcular os descritores de Haralick
def calculate_haralick_features(image):
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    image_uint8 = img_as_ubyte(image)
    glcm = graycomatrix(image_uint8, distances=distances, angles=angles, symmetric=True, normed=True)
    feature_vector = []
    for prop in properties:
        for angle in range(len(angles)):
            feature = graycoprops(glcm, prop)[0, angle]
            feature_vector.append(feature)
    return feature_vector

# Função para pré-processar a imagem e calcular características de Haralick
def preprocess_image_and_extract_features(image_path):
    # Carregar a imagem
    image = io.imread(image_path)
    
    # Calcular características de Haralick
    haralick_features = calculate_haralick_features(image[..., 0])  # Usar apenas o primeiro canal
    
    # Pré-processamento específico para ResNet50
    if len(image.shape) == 2:  # Se a imagem for grayscale
        image = np.expand_dims(image, axis=-1)  # Para tornar a imagem 3 canais
        image = np.repeat(image, 3, axis=-1)  # Repetir o canal único 3 vezes
    image = np.resize(image, (224, 224, 3))  # Redimensionar para (224, 224, 3)
    image = preprocess_input(image)  # Pré-processamento ResNet50
    
    return image, haralick_features

# Função para processar todas as imagens em subpastas
def process_images_in_directory():
    predictions = []
    
    # Percorrer todas as subpastas
    for subdir, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png')):  # Ajuste conforme necessário
                image_path = os.path.join(subdir, file)
                
                # Pré-processar a imagem e extrair características de Haralick
                image, haralick_features = preprocess_image_and_extract_features(image_path)
                
                # Fazer a previsão usando o modelo ResNet50 para 6 classes
                prediction = MODEL.predict([np.expand_dims(image, axis=0), np.expand_dims(haralick_features, axis=0)])
                predicted_class = np.argmax(prediction, axis=1)[0]  # Obter a classe predita

                if predicted_class == 0:
                    pred = 'Negative for intraepithelial lesion'
                elif predicted_class == 1:
                    pred = 'SCC'
                elif predicted_class == 2:
                    pred = 'LSIL'
                elif predicted_class == 3:
                    pred = 'ASC-H'
                elif predicted_class == 4:
                    pred = 'HSIL'
                elif predicted_class == 5:
                    pred = 'ASC-US'


                # Armazenar o resultado
                predictions.append((image_path, pred))
    
    label_map = {
        0: 'Classe 0',
        1: 'Classe 1',
        2: 'Classe 2',
        3: 'Classe 3',
        4: 'Classe 4',
        5: 'Classe 5'
    }

    return predictions
    # Imprimir os resultados
    print("\nResultados da Classificação de Imagens:\n")
    for image_path, predicted_class in predictions:
        predicted_label = label_map.get(predicted_class, 'Classe Desconhecida')
        print(f'Imagem: {image_path} -> Previsão: {predicted_label}')
