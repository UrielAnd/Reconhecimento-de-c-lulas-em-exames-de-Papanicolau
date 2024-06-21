import os
import glob
import numpy as np
from skimage import io, img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nucleus_detection

# Diretório onde as imagens estão armazenadas
image_dir = 'image_nucleus'
diretorio = 'cell_images'

# Mapeamento de subpastas para rótulos binários
label_map = {
    'SCC': 1,
    'LSIL': 1,
    'ASC-H': 1,
    'HSIL': 1,
    'ASC-US': 1,
    'Negative for intraepithelial lesion': 0
}

# Placeholder para os dados e rótulos
data = []
labels = []
haralick_features = []

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

# Obter características das imagens usando a função do módulo nucleus_detection
nomes_arquivos = os.listdir(diretorio)
for nome in nomes_arquivos:
    nucleus_detection.get_characteristics("cell_images\\" + nome)

# Ler todas as imagens das subpastas e atribuir rótulos
for subfolder, label in label_map.items():
    subfolder_path = os.path.join(image_dir, subfolder)
    image_paths = glob.glob(os.path.join(subfolder_path, '*.png'))  # Ajuste a extensão conforme necessário

    for image_path in image_paths:
        image = io.imread(image_path)
        if len(image.shape) == 2:  # Se a imagem for grayscale
            image = np.expand_dims(image, axis=-1)  # Para tornar a imagem 3 canais
            image = np.repeat(image, 3, axis=-1)  # Repetir o canal único 3 vezes
        image = np.resize(image, (224, 224, 3))  # Redimensionar para (224, 224, 3)
        data.append(image)
        labels.append(label)

        # Calcular e armazenar características de Haralick
        gray_image = img_as_ubyte(image[..., 0])  # Usar apenas o primeiro canal para a conversão em grayscale
        haralick_features.append(calculate_haralick_features(gray_image))

# Convertendo para array numpy
data = np.array(data)
labels = np.array(labels)
haralick_features = np.array(haralick_features)

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test, haralick_train, haralick_test = train_test_split(
    data, labels, haralick_features, test_size=0.2, random_state=42, stratify=labels)

# One-hot encode os rótulos
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Carregar o modelo ResNet50 pré-treinado
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar todas as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# Adicionar camadas personalizadas no topo do modelo base
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Adicionar a entrada de características de Haralick
haralick_input = Input(shape=(24,), name='haralick_input')  # 6 propriedades * 4 ângulos
concatenated = concatenate([x, haralick_input])

# Adicionar camadas densas
x = Dense(1024, activation='relu')(concatenated)
predictions = Dense(2, activation='softmax')(x)

# Criar o modelo final
model = Model(inputs=[base_model.input, haralick_input], outputs=predictions)

# Compilar o modelo
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Definir callbacks
checkpoint = ModelCheckpoint('resnet50_best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Treinar o modelo
history = model.fit([X_train, haralick_train], y_train, validation_data=([X_test, haralick_test], y_test),
                    epochs=50, batch_size=32, callbacks=[checkpoint, early_stopping])

# Avaliar o modelo no conjunto de teste
loss, accuracy = model.evaluate([X_test, haralick_test], y_test)
print(f"Acurácia: {accuracy}")

# Predizer os rótulos no conjunto de teste
y_pred = model.predict([X_test, haralick_test])
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calcular a matriz de confusão
cm = confusion_matrix(y_true, y_pred_classes)

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Valores Preditos')
plt.ylabel('Valores Reais')
plt.title('Matriz de Confusão')
plt.show()

# Plotar as curvas de aprendizado
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Teste')
plt.title('Acurácia durante o Treinamento e Teste')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Teste')
plt.title('Perda durante o Treinamento e Teste')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()

plt.show()
