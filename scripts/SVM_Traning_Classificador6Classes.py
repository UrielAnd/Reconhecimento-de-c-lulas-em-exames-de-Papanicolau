import os
import glob
import numpy as np
from skimage import io
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump
import nucleus_detection
import matplotlib.pyplot as plt
import seaborn as sns

# Diretório onde as imagens estão armazenadas
image_dir = 'image_nucleus'

diretorio = 'cell_images'
# Mapeamento de subpastas para rótulos com 6 classes
label_map = {
    'SCC': 1,
    'LSIL': 2,
    'ASC-H': 3,
    'HSIL': 4,
    'ASC-US': 5,
    'Negative for intraepithelial lesion': 0
}

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

# Placeholder para os dados e rótulos
data = []
labels = []
nomes_arquivos = os.listdir(diretorio)
for nome in nomes_arquivos:
    nucleus_detection.get_characteristics("cell_images\\"+nome)

# Ler todas as imagens das subpastas e atribuir rótulos
for subfolder, label in label_map.items():
    subfolder_path = os.path.join(image_dir, subfolder)
    image_paths = glob.glob(os.path.join(subfolder_path, '*.png'))  # Ajuste a extensão conforme necessário

    for image_path in image_paths:
        image = io.imread(image_path, as_gray=True)
        features = calculate_haralick_features(image)
        data.append(features)
        labels.append(label)

# Convertendo para array numpy
data = np.array(data)
labels = np.array(labels)

# Dividir os dados em conjuntos de treino e teste com estratificação
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

# Treinar o classificador SVM
classifier = svm.SVC(kernel='linear', C=1)  # Parâmetros podem ser ajustados
classifier.fit(X_train, y_train)

# Prever no conjunto de teste
y_pred = classifier.predict(X_test)

# Calcular a acurácia
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Acurácia: {accuracy}")
print(f"Classification Report:\n{report}")

# Calcular a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_map.keys(), yticklabels=label_map.keys())
plt.xlabel('Valores Preditos')
plt.ylabel('Valores Reais')
plt.title('Matriz de Confusão')
plt.show()

# Plotar a acurácia em um gráfico de barras
plt.figure(figsize=(6, 4))
plt.bar(['Accuracy'], [accuracy], color='blue')
plt.ylabel('Acurácia')
plt.title('Acurácia do Classificador')
plt.ylim(0, 1)  # Limitar o eixo y de 0 a 1 para a escala de acurácia
plt.show()