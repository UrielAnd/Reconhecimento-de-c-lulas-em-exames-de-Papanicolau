import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mahotas
import nucleus_detection
from skimage.feature import graycomatrix, graycoprops
from skimage import exposure, color, io
from skimage.measure import moments_hu

def selecionar_imagem():
    global caminho_arquivo, imagem, zoom_factor, Img_conv
    Img_conv = False
    caminho_arquivo = filedialog.askopenfilename(
        filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.gif")]
    )
    if caminho_arquivo:
        imagem = Image.open(caminho_arquivo)
        imagem.thumbnail((400, 400))  # Redimensiona a imagem para caber na tela
        zoom_factor = 1.0
        exibir_imagem(imagem)
        nucleus_detection.get_characteristics(caminho_arquivo)
        label_caminho.config(text=caminho_arquivo)
        botao_tons_cinza.config(fg="blue",cursor="hand2", relief="raised")
        botao_gerar_histograma.config(fg="blue",cursor="hand2", relief="raised")
        botao_zoom_in.config(fg="blue",cursor="hand2", relief="raised")
        botao_zoom_out.config(fg="blue",cursor="hand2", relief="raised")
        botao_gerar_histograma_hiv.config(fg="blue",cursor="hand2", relief="raised")

def exibir_imagem(imagem):
    global zoom_factor
    img_resized = imagem.resize((int(imagem.width * zoom_factor), int(imagem.height * zoom_factor)), Image.LANCZOS)
    imagem_tk = ImageTk.PhotoImage(img_resized)
    if Img_conv:
        label_imagem_convertida.config(image=imagem_tk)
        label_imagem_convertida.image = imagem_tk
        
    else:
        label_imagem.config(image=imagem_tk)
        label_imagem.image = imagem_tk
        label_imagem_convertida.pack_forget()
        label_imagem.pack_forget()
        label_imagem.pack()
        frame_botoes_histogramas.pack_forget()
        frame_botoes_histogramas.pack()
        botalharalick.pack_forget()
        botalharalick.pack()
        botaoHu.pack_forget()
        botaoHu.pack()
        botao_gerar_mat_coocorrencia.pack_forget()
        botao_gerar_mat_coocorrencia.pack()
        botao_gerar_inv_hu_256.pack_forget()
        botao_gerar_inv_hu_256.pack()
        botao_tons_cinza.config(text="Exibir imagem em tons de cinza", fg="blue", relief="raised")
        botalharalick.config(fg="gray", cursor="arrow", relief="sunken")
        botaoHu.config(fg="gray", cursor="arrow", relief="sunken")
        botao_gerar_histograma_16.config(fg="gray", cursor="arrow", relief="sunken")
        botao_gerar_mat_coocorrencia.config(fg="gray", cursor="arrow", relief="sunken")
        botao_gerar_inv_hu_256.config(fg="gray", cursor="arrow", relief="sunken")


def zoom_in():
    global zoom_factor, imagem, imagem_cinza
    if Img_conv:
        zoom_factor *= 1.1
        exibir_imagem(imagem_cinza)
    else:
        zoom_factor *= 1.1
        exibir_imagem(imagem)

def zoom_out():
    global zoom_factor, imagem, imagem_cinza
    if Img_conv:
        zoom_factor /= 1.1
        exibir_imagem(imagem_cinza)
    else:
        zoom_factor /= 1.1
        exibir_imagem(imagem)

def converter_img():
    global Img_conv, Img_normal, imagem_cinza
    if Img_conv:
        #-----------------------------------------#
        #Variáveis de controle
        Img_normal = True
        Img_conv = False
        #-----------------------------------------#
        exibir_imagem(imagem)
        botao_tons_cinza.config(text="Exibir imagem em tons de cinza", fg="blue", relief="raised")
        botalharalick.config(fg="gray", cursor="arrow", relief="sunken")
        botaoHu.config(fg="gray", cursor="arrow", relief="sunken")
        botao_gerar_histograma_16.config(fg="gray", cursor="arrow", relief="sunken")
        botao_gerar_mat_coocorrencia.config(fg="gray", cursor="arrow", relief="sunken")
        botao_gerar_inv_hu_256.config(fg="gray", cursor="arrow", relief="sunken")
        #-----------------------------------------#
        #Re organizando labels
        label_imagem.pack()
        
        frame_botoes_histogramas.pack_forget()
        frame_botoes_histogramas.pack()
        botalharalick.pack_forget()
        botalharalick.pack()
        botaoHu.pack_forget()
        botaoHu.pack()
        botao_gerar_mat_coocorrencia.pack_forget()
        botao_gerar_mat_coocorrencia.pack()
        botao_gerar_inv_hu_256.pack_forget()
        botao_gerar_inv_hu_256.pack()
        #-----------------------------------------#
    else:
        imagem_cinza = imagem.convert('L') # Converte para tons de cinza
        img_resized = imagem_cinza.resize((int(imagem.width * zoom_factor), int(imagem.height * zoom_factor)), Image.LANCZOS)
        imagem_cinza_tk = ImageTk.PhotoImage(img_resized)
        label_imagem_convertida.config(image=imagem_cinza_tk)
        label_imagem_convertida.image = imagem_cinza_tk
        botao_tons_cinza.config(text="Ocultar imagem em tons de cinza")
        botalharalick.config(fg="blue", cursor="hand2", relief="raised")
        botaoHu.config(fg="blue", cursor="hand2", relief="raised")
        botao_gerar_histograma_16.config(fg="blue", cursor="hand2", relief="raised")
        botao_gerar_histograma_hiv.config(fg="gray", cursor="arrow", relief="sunken")
        botao_gerar_mat_coocorrencia.config(fg="blue", cursor="hand2", relief="raised")
        botao_gerar_inv_hu_256.config(fg="blue", cursor="hand2", relief="raised")
        #-----------------------------------------#
        #Re organizando labels
        label_imagem.pack_forget()
        label_imagem_convertida.pack()
        frame_botoes_histogramas.pack_forget()
        frame_botoes_histogramas.pack()
        botalharalick.pack_forget()
        botalharalick.pack()
        botaoHu.pack_forget()
        botaoHu.pack()
        botao_gerar_mat_coocorrencia.pack_forget()
        botao_gerar_mat_coocorrencia.pack()
        botao_gerar_inv_hu_256.pack_forget()
        botao_gerar_inv_hu_256.pack()
        #-----------------------------------------#
        #-----------------------------------------#
        #Variáveis de controle
        Img_conv = True
        Img_normal = False
        #-----------------------------------------#
def gerar_histograma():
    global Img_normal, Img_conv
    if Img_normal:
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
    elif Img_conv:
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
    else:
        ...


def gerar_histograma_16tons():
    if Img_conv:
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

def gerar_histograma_hiv():
        if not Img_conv: 
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

def calcular_descritores_haralick():
    global Img_conv
    if Img_conv:
        # Abre a imagem e converte para tons de cinza
        imagem = Image.open(caminho_arquivo).convert('L')
        # Converte a imagem para um array numpy para processamento
        imagem_np = np.array(imagem)
        # Calcula os descritores de Haralick
        descritores = mahotas.features.haralick(imagem_np)
        # Calcula a média dos descritores para cada direção
        descritores_medios = descritores.mean(axis=0)
        popup = tk.Toplevel()
        popup.title("Descritores_haralick")
        tk.Label(popup, text=descritores_medios).pack(pady=10)
        tk.Button(popup, text="Fechar", command=popup.destroy).pack(pady=10)
        print(descritores)
    else:
        label_error.config(text="ERRO: Para que esse modulo funcione você precisa alterar a imagem para preto e branco!!!", fg="red")

def calcular_momentos_hu():
    global Img_conv
    if Img_conv:
        # Carrega a imagem
        imagem = cv2.imread(caminho_arquivo)
        # Converte a imagem para tons de cinza
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        # Calcula os momentos da imagem
        momentos = cv2.moments(imagem_cinza)
        # Calcula os momentos invariantes de Hu a partir dos momentos
        momentos_hu = cv2.HuMoments(momentos)
        # Log transform para facilitar a visualização
        momentos_hu_log = -np.sign(momentos_hu) * np.log10(np.abs(momentos_hu))
        popup = tk.Toplevel()
        popup.title("Descritores_haralick")
        tk.Label(popup, text=momentos_hu_log).pack(pady=10)
        tk.Button(popup, text="Fechar", command=popup.destroy).pack(pady=10)
    else:
        label_error.config(text="ERRO: Para que esse modulo funcione você precisa alterar a imagem para preto e branco!!!", fg="red")
def calcular_descritores_matriz_coocorencia(glcm):
    # Contraste
    contraste = graycoprops(glcm, 'contrast')[0, 0]
    
    # Homogeneidade
    homogeneidade = graycoprops(glcm, 'homogeneity')[0, 0]
    
    # Entropia
    with np.errstate(divide='ignore', invalid='ignore'):
        glcm_normalized = glcm / np.sum(glcm)
        entropy = -np.sum(glcm_normalized * np.log2(glcm_normalized + (glcm_normalized == 0)))
    
    return contraste, homogeneidade, entropy

def gerar_matriz_coocorrencia():

    if Img_conv:
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
            contraste, homogeneidade, entropia = calcular_descritores_matriz_coocorencia(glcm)
            descritores.append((contraste, homogeneidade, entropia))

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

def gerar_invariantes_hu_256():
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



# Configuração da janela principal
janela = tk.Tk()
janela.title("Papanicolau")
janela.geometry("800x1000")
frame_botoes_zoom = tk.Frame(janela)
frame_botoes_histogramas= tk.Frame(janela)
#-------------------------#
#Variáveis controladoras de estado
imagem_convertida = True
Img_normal = True
Img_conv = False
imagem_cinza = None
#-------------------------#
#---------------------------------------------------#
#Labels
# Botão para selecionar a imagem
botao_selecionar = tk.Button(janela, text="Selecionar Imagem", command=selecionar_imagem, fg="blue", cursor="hand2")
# Label para exibir o caminho da imagem selecionada
label_caminho = tk.Label(janela, text="")
# Label para exibir a imagem selecionada
label_imagem = tk.Label(janela)
label_imagem_convertida = tk.Label(janela)
botao_tons_cinza = tk.Button(janela, text="Exibir imagem em tons de cinza", command=converter_img, height=1, width=30, fg="gray", relief="sunken")
botao_gerar_histograma = tk.Button(frame_botoes_histogramas, text="Gerar", command=gerar_histograma, height=2, width=15, fg="gray", relief="sunken")
botao_gerar_histograma_16 = tk.Button(frame_botoes_histogramas, text="16 tons", command=gerar_histograma_16tons, height=2, width=15, fg="gray", relief="sunken")
botao_gerar_histograma_hiv = tk.Button(frame_botoes_histogramas, text="2D HV", command=gerar_histograma_hiv, height=2, width=15, fg="gray", relief="sunken")
botalharalick = tk.Button(janela, text="Mostrar descritores de Haralick", command=calcular_descritores_haralick, height=2, width=30, fg="gray", relief="sunken")
botaoHu = tk.Button(janela, text="Mostrar invariantes de HU", command=calcular_momentos_hu, height=2, width=30, fg="gray", relief="sunken")
botao_gerar_mat_coocorrencia = tk.Button(janela, text="Gerar matriz de co-ocorrência", command=gerar_matriz_coocorrencia, height=2, width=30, fg="gray", relief="sunken")
botao_gerar_inv_hu_256 = tk.Button(janela, text="Invariantes de Hu 256", command=gerar_invariantes_hu_256, height=2, width=30, fg="gray", relief="sunken")
label_error = tk.Label(janela)
# Botões de zoom
botao_zoom_in = tk.Button(frame_botoes_zoom, text="➕", command=zoom_in, height=1, width=5, fg="gray", relief="sunken")
botao_zoom_out = tk.Button(frame_botoes_zoom, text="➖", command=zoom_out, height=1, width=5, fg="gray", relief="sunken")

#---------------------------------------------------#
#---------------------------------------------------#
#Plotar labels na tela
botao_selecionar.pack(pady=3)
botao_tons_cinza.pack()
label_caminho.pack()
frame_botoes_zoom.pack()
botao_zoom_in.pack(side="left")
botao_zoom_out.pack(side="left")
label_imagem.pack()
frame_botoes_histogramas.pack()
botao_gerar_histograma.pack(padx=6, pady=6, side="left")
botao_gerar_histograma_16.pack(padx=6, pady=6, side="left")
botao_gerar_histograma_hiv.pack(padx=6, pady=6, side="left")
botalharalick.pack(padx=6, pady=6)
botaoHu.pack(padx=6, pady=6)
botao_gerar_mat_coocorrencia.pack(padx=6, pady=6)
label_error.pack()
botao_gerar_inv_hu_256.pack()
#---------------------------------------------------#

# Executa a aplicação
janela.mainloop()
