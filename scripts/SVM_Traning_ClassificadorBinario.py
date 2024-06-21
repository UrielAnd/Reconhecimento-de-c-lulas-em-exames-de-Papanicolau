import os
import glob
import numpy as np
from skimage import io
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump
import nucleus_detection

# Diretório onde as imagens estão armazenadas
image_dir = 'image_nucleus'
diretorio = 'cell_images'

# Mapeamento de subpastas para rótulos
label_map = {
    'SCC': 1,
    'LSIL': 1,
    'ASC-H': 1,
    'HSIL': 1,
    'ASC-US': 1,
    'Negative for intraepithelial lesion': 0
}

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

# Placeholder para os dados e rótulos
data = []
labels = []
nomes_arquivos = os.listdir(diretorio)
for nome in nomes_arquivos:
    nucleus_detection.get_characteristics("cell_images\\"+nome)

# Ler todas as imagens das subpastas e atribuir rótulos
for subfolder, label in label_map.items():
    subfolder_path = os.path.join(image_dir, subfolder)
    image_paths = glob.glob(os.path.join(subfolder_path, '*.png'))
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
classifier = svm.SVC(kernel='linear', C=1)
classifier.fit(X_train, y_train)

# Salvar o modelo treinado
model_filename = 'svm_model.joblib'
dump(classifier, model_filename)

# Prever no conjunto de teste
y_pred = classifier.predict(X_test)

# Avaliar o classificador
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

# Exibir a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Plotar a acurácia do treinamento
results = {
    'Accuracy': accuracy
}

# Exibir a tabela de resultados
results_table = {
    'Metric': ['Accuracy'],
    'Value': [accuracy]
}

print("\nTraining Results")
for key, value in results_table.items():
    print(f"{key}: {value}")

# Plotar a acurácia
plt.figure(figsize=(8, 6))
plt.bar(results_table['Metric'], results_table['Value'], color='blue')
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Training Accuracy')
plt.show()