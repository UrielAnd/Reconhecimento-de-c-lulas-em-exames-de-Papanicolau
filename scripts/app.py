import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import nucleus_detection
import histograma
import haralick
import hu
import matriz_coocorrencia
import SVM_Use_classificadorBinario
import SVM_Use_classificador6Classes
from tkinter import ttk
import RESNET50_Use_Classificador6Classes
import RESNET50_Use_ClassificadorBinario
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
        botao_gerar_inv_hu_256.config(fg="blue", cursor="hand2", relief="raised")
        botao_SVM_6_Class_predict.config(fg="blue", cursor="hand2", relief="raised")
        botao_SVM_Binary_predict.config(fg="blue", cursor="hand2", relief="raised")
        botao_RESNET50_6_Class_predict.config(fg="blue", cursor="hand2", relief="raised")
        botao_RESNET50_Binary_predict.config(fg="blue", cursor="hand2", relief="raised")
        botao_acuracia_resnet.config(fg="blue", cursor="hand2", relief="raised")
        botao_matriz_resnet.config(fg="blue", cursor="hand2", relief="raised")
        botao_acuracia_svm.config(fg="blue", cursor="hand2", relief="raised")
        botao_matriz_svm.config(fg="blue", cursor="hand2", relief="raised")
def exibir_imagem(imagem):
    global zoom_factor
    img_resized = imagem.resize((int(imagem.width * zoom_factor), int(imagem.height * zoom_factor)), Image.LANCZOS)
    imagem_tk = ImageTk.PhotoImage(img_resized)
    if Img_conv:
        label_imagem_convertida.config(image=imagem_tk)
        label_imagem_convertida.image = imagem_tk
        
    else:
        #-----------------------------------------#
        #Re organizando labels
        label_imagem.config(image=imagem_tk)
        label_imagem.image = imagem_tk
        label_imagem_convertida.pack_forget()
        label_imagem.pack_forget()
        label_imagem.pack()
        label_histograma.pack_forget()
        label_histograma.pack()
        frame_botoes_histogramas.pack_forget()
        frame_botoes_histogramas.pack()
        label_hu.pack_forget()
        label_hu.pack()
        frame_botoes_hu.pack_forget()
        frame_botoes_hu.pack()
        label_haralick.pack_forget()
        label_haralick.pack
        botalharalick.pack_forget()
        botalharalick.pack()
        label_matriz.pack_forget()
        label_matriz.pack()
        botao_gerar_mat_coocorrencia.pack_forget()
        botao_gerar_mat_coocorrencia.pack()
        label_SVM.pack_forget()
        label_SVM.pack()
        botao_SVM_Binary_predict.pack_forget()
        botao_SVM_Binary_predict.pack()
        botao_SVM_6_Class_predict.pack_forget()
        botao_SVM_6_Class_predict.pack()
        botao_acuracia_svm.pack_forget()
        botao_matriz_svm.pack_forget()
        botao_acuracia_svm.pack()
        botao_matriz_svm.pack()
        label_RESNET50.pack_forget()
        botao_RESNET50_Binary_predict.pack_forget()
        botao_RESNET50_6_Class_predict.pack_forget()
        label_RESNET50.pack()
        botao_RESNET50_Binary_predict.pack()
        botao_RESNET50_6_Class_predict.pack()
        botao_acuracia_resnet.pack_forget()
        botao_matriz_resnet.pack_forget()
        botao_acuracia_resnet.pack()
        botao_matriz_resnet.pack()
        #-----------------------------------------#
        botao_tons_cinza.config(text="Exibir imagem em tons de cinza", fg="blue", relief="raised")
        botalharalick.config(fg="gray", cursor="arrow", relief="sunken")
        botaoHu.config(fg="blue", cursor="hand2", relief="raised")
        botao_gerar_histograma_16.config(fg="gray", cursor="arrow", relief="sunken")
        botao_gerar_mat_coocorrencia.config(fg="gray", cursor="arrow", relief="sunken")
        

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
        botao_gerar_histograma_16.config(fg="gray", cursor="arrow", relief="sunken")
        botao_gerar_mat_coocorrencia.config(fg="gray", cursor="arrow", relief="sunken")
        #-----------------------------------------#
        #Re organizando labels
        label_imagem.pack()
        
        label_histograma.pack_forget()
        label_histograma.pack()
        frame_botoes_histogramas.pack_forget()
        frame_botoes_histogramas.pack()
        label_hu.pack_forget()
        label_hu.pack()
        frame_botoes_hu.pack_forget()
        frame_botoes_hu.pack()
        label_haralick.pack_forget()
        label_haralick.pack
        botalharalick.pack_forget()
        botalharalick.pack()
        label_matriz.pack_forget()
        label_matriz.pack()
        botao_gerar_mat_coocorrencia.pack_forget()
        botao_gerar_mat_coocorrencia.pack()
        label_SVM.pack_forget()
        label_SVM.pack()
        botao_SVM_Binary_predict.pack_forget()
        botao_SVM_Binary_predict.pack()
        botao_SVM_6_Class_predict.pack_forget()
        botao_SVM_6_Class_predict.pack()
        botao_acuracia_svm.pack_forget()
        botao_matriz_svm.pack_forget()
        botao_acuracia_svm.pack()
        botao_matriz_svm.pack()
        label_RESNET50.pack_forget()
        botao_RESNET50_Binary_predict.pack_forget()
        botao_RESNET50_6_Class_predict.pack_forget()
        label_RESNET50.pack()
        botao_RESNET50_Binary_predict.pack()
        botao_RESNET50_6_Class_predict.pack()
        botao_acuracia_resnet.pack_forget()
        botao_matriz_resnet.pack_forget()
        botao_acuracia_resnet.pack()
        botao_matriz_resnet.pack()
        #-----------------------------------------#
    else:
        imagem_cinza = imagem.convert('L') # Converte para tons de cinza
        img_resized = imagem_cinza.resize((int(imagem.width * zoom_factor), int(imagem.height * zoom_factor)), Image.LANCZOS)
        imagem_cinza_tk = ImageTk.PhotoImage(img_resized)
        label_imagem_convertida.config(image=imagem_cinza_tk)
        label_imagem_convertida.image = imagem_cinza_tk
        botao_tons_cinza.config(text="Ocultar imagem em tons de cinza")
        botalharalick.config(fg="blue", cursor="hand2", relief="raised")
        botao_gerar_histograma_16.config(fg="blue", cursor="hand2", relief="raised")
        botao_gerar_histograma_hiv.config(fg="gray", cursor="arrow", relief="sunken")
        botao_gerar_mat_coocorrencia.config(fg="blue", cursor="hand2", relief="raised")
        #-----------------------------------------#
        #Re organizando labels
        label_imagem.pack_forget()
        label_imagem_convertida.pack()
        label_histograma.pack_forget()
        label_histograma.pack()
        frame_botoes_histogramas.pack_forget()
        frame_botoes_histogramas.pack()
        label_hu.pack_forget()
        label_hu.pack()
        frame_botoes_hu.pack_forget()
        frame_botoes_hu.pack()
        label_haralick.pack_forget()
        label_haralick.pack
        botalharalick.pack_forget()
        botalharalick.pack()
        label_matriz.pack_forget()
        label_matriz.pack()
        botao_gerar_mat_coocorrencia.pack_forget()
        botao_gerar_mat_coocorrencia.pack()
        label_SVM.pack_forget()
        label_SVM.pack()
        botao_SVM_Binary_predict.pack_forget()
        botao_SVM_Binary_predict.pack()
        botao_SVM_6_Class_predict.pack_forget()
        botao_SVM_6_Class_predict.pack()
        botao_acuracia_svm.pack_forget()
        botao_matriz_svm.pack_forget()
        botao_acuracia_svm.pack()
        botao_matriz_svm.pack()
        label_RESNET50.pack_forget()
        botao_RESNET50_Binary_predict.pack_forget()
        botao_RESNET50_6_Class_predict.pack_forget()
        label_RESNET50.pack()
        botao_RESNET50_Binary_predict.pack()
        botao_RESNET50_6_Class_predict.pack()
        botao_acuracia_resnet.pack_forget()
        botao_matriz_resnet.pack_forget()
        botao_acuracia_resnet.pack()
        botao_matriz_resnet.pack()
        #-----------------------------------------#
        #-----------------------------------------#
        #Variáveis de controle
        Img_conv = True
        Img_normal = False
        #-----------------------------------------#
def SVMBiPred():
    results = SVM_Use_classificadorBinario.process_images_in_directory()
    result_window = tk.Toplevel(janela)
    result_window.title("Resultados da Classificação de Imagens")
    result_window.geometry("500x800")
    # Frame para exibir a tabela com scrollbar
    result_frame = ttk.Frame(result_window)
    result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Treeview para exibir os resultados como uma tabela
    tree = ttk.Treeview(result_frame, columns=("Nome do Arquivo", "Previsão"), show="headings")
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Configuração das colunas
    tree.heading("Nome do Arquivo", text="Nome do Arquivo")
    tree.heading("Previsão", text="Previsão")
    
    # Definindo o alinhamento central para todas as colunas
    tree.column("Nome do Arquivo", anchor=tk.CENTER)
    tree.column("Previsão", anchor=tk.CENTER)
    
    # Adiciona uma barra de rolagem ao treeview
    scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=tree.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tree.configure(yscrollcommand=scrollbar.set)
    
    # Adiciona os resultados ao treeview
    for idx, (image_path, prediction) in enumerate(results):
        filename = os.path.basename(image_path)  # Obtém apenas o nome do arquivo da imagem
        tree.insert("", "end", values=(filename, f"Classe {prediction}"))
    
    # Botão para fechar a janela de resultados
    close_button = ttk.Button(result_window, text="Fechar", command=result_window.destroy)
    close_button.pack(pady=10)

    # Imprimir os resultados
    # for image_path, prediction in results:
    #     print(f'Imagem: {image_path} - Previsão: {prediction}')
def SVM6ClassPred():
    results = SVM_Use_classificador6Classes.process_images_in_directory()
    # Criação da janela de resultados
    result_window = tk.Toplevel()
    result_window.title("Resultados da Classificação de Imagens")
    result_window.geometry("500x800")
    # Frame para exibir a tabela com scrollbar
    result_frame = ttk.Frame(result_window)
    result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Treeview para exibir os resultados como uma tabela
    tree = ttk.Treeview(result_frame, columns=("Nome do Arquivo", "Previsão"), show="headings")
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Configuração das colunas
    tree.heading("Nome do Arquivo", text="Nome do Arquivo")
    tree.heading("Previsão", text="Previsão")
    
    # Definindo o alinhamento central para todas as colunas
    tree.column("Nome do Arquivo", anchor=tk.CENTER)
    tree.column("Previsão", anchor=tk.CENTER)
    
    # Adiciona uma barra de rolagem ao treeview
    scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=tree.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tree.configure(yscrollcommand=scrollbar.set)
    
    # Adiciona os resultados ao treeview
    for image_path, prediction in results:
        filename = os.path.basename(image_path)  # Obtém apenas o nome do arquivo da imagem
        tree.insert("", "end", values=(filename, f"Classe {prediction}"))
    
    # Botão para fechar a janela de resultados
    close_button = ttk.Button(result_window, text="Fechar", command=result_window.destroy)
    close_button.pack(pady=10)
def RESNET50BiPred():
    results = RESNET50_Use_ClassificadorBinario.process_images_in_directory()
    result_window = tk.Toplevel(janela)
    result_window.title("Resultados da Classificação de Imagens")
    result_window.geometry("500x800")
    # Frame para exibir a tabela com scrollbar
    result_frame = ttk.Frame(result_window)
    result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Treeview para exibir os resultados como uma tabela
    tree = ttk.Treeview(result_frame, columns=("Nome do Arquivo", "Previsão"), show="headings")
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Configuração das colunas
    tree.heading("Nome do Arquivo", text="Nome do Arquivo")
    tree.heading("Previsão", text="Previsão")
    
    # Definindo o alinhamento central para todas as colunas
    tree.column("Nome do Arquivo", anchor=tk.CENTER)
    tree.column("Previsão", anchor=tk.CENTER)
    
    # Adiciona uma barra de rolagem ao treeview
    scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=tree.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tree.configure(yscrollcommand=scrollbar.set)
    
    # Adiciona os resultados ao treeview
    for idx, (image_path, prediction) in enumerate(results):
        filename = os.path.basename(image_path)  # Obtém apenas o nome do arquivo da imagem
        tree.insert("", "end", values=(filename, f"Classe {prediction}"))
    
    # Botão para fechar a janela de resultados
    close_button = ttk.Button(result_window, text="Fechar", command=result_window.destroy)
    close_button.pack(pady=10)

def RESNET506classPred():
    results = RESNET50_Use_Classificador6Classes.process_images_in_directory()
    # Criação da janela de resultados
    result_window = tk.Toplevel()
    result_window.title("Resultados da Classificação de Imagens")
    result_window.geometry("500x800")
    # Frame para exibir a tabela com scrollbar
    result_frame = ttk.Frame(result_window)
    result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Treeview para exibir os resultados como uma tabela
    tree = ttk.Treeview(result_frame, columns=("Nome do Arquivo", "Previsão"), show="headings")
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Configuração das colunas
    tree.heading("Nome do Arquivo", text="Nome do Arquivo")
    tree.heading("Previsão", text="Previsão")
    
    # Definindo o alinhamento central para todas as colunas
    tree.column("Nome do Arquivo", anchor=tk.CENTER)
    tree.column("Previsão", anchor=tk.CENTER)
    
    # Adiciona uma barra de rolagem ao treeview
    scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=tree.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tree.configure(yscrollcommand=scrollbar.set)
    
    # Adiciona os resultados ao treeview
    for image_path, prediction in results:
        filename = os.path.basename(image_path)  # Obtém apenas o nome do arquivo da imagem
        tree.insert("", "end", values=(filename, f"Classe {prediction}"))
    
    # Botão para fechar a janela de resultados
    close_button = ttk.Button(result_window, text="Fechar", command=result_window.destroy)
    close_button.pack(pady=10)


def abrir_imagem_svm():
    # Cria uma nova janela
    nova_janela = tk.Toplevel(janela)
    nova_janela.title("Imagens")

    # Caminhos das imagens
    caminho_imagem1 = "img//SVM_Binarioacc.png"
    caminho_imagem2 = "img//SVM6classacc.png"


    # Carrega as imagens
    img1 = Image.open(caminho_imagem1)
    img1 = ImageTk.PhotoImage(img1.resize((300, 300)))  # Resize the image to desired dimensions
    img2 = Image.open(caminho_imagem2)
    img2 = ImageTk.PhotoImage(img2.resize((300, 300)))  # Resize the image to desired dimensions

    # Cria rótulos para exibir as imagens
    label1 = tk.Label(nova_janela, image=img1)
    label1.image = img1  # Armazena a referência da imagem
    label1.pack(side="left", padx=10, pady=10)

    label2 = tk.Label(nova_janela, image=img2)
    label2.image = img2  # Armazena a referência da imagem
    label2.pack(side="left", padx=10, pady=10)


def abrir_imagem_resnet():
    # Cria uma nova janela
    nova_janela = tk.Toplevel(janela)
    nova_janela.title("Imagens")

    # Caminhos das imagens
    caminho_imagem1 = "img//ResnetBinarioacc.png"
    caminho_imagem2 = "img//Resnet6Classacc.png"


    # Carrega as imagens
    img1 = Image.open(caminho_imagem1)
    img1 = ImageTk.PhotoImage(img1.resize((600, 300)))  # Resize the image to desired dimensions
    img2 = Image.open(caminho_imagem2)
    img2 = ImageTk.PhotoImage(img2.resize((600, 300)))  # Resize the image to desired dimensions

    # Cria rótulos para exibir as imagens
    label1 = tk.Label(nova_janela, image=img1)
    label1.image = img1  # Armazena a referência da imagem
    label1.pack(side="left", padx=10, pady=10)

    label2 = tk.Label(nova_janela, image=img2)
    label2.image = img2  # Armazena a referência da imagem
    label2.pack(side="left", padx=10, pady=10)


def abrir_imagem_matriz_svm():
    # Cria uma nova janela
    nova_janela = tk.Toplevel(janela)
    nova_janela.title("Imagens")

    # Caminhos das imagens
    caminho_imagem1 = "img//MatrizBiSVM.png"
    caminho_imagem2 = "img//Matriz6classSVM.png"


    # Carrega as imagens
    img1 = Image.open(caminho_imagem1)
    img1 = ImageTk.PhotoImage(img1.resize((300, 300)))  # Resize the image to desired dimensions
    img2 = Image.open(caminho_imagem2)
    img2 = ImageTk.PhotoImage(img2.resize((300, 300)))  # Resize the image to desired dimensions

    # Cria rótulos para exibir as imagens
    label1 = tk.Label(nova_janela, image=img1)
    label1.image = img1  # Armazena a referência da imagem
    label1.pack(side="left", padx=10, pady=10)

    label2 = tk.Label(nova_janela, image=img2)
    label2.image = img2  # Armazena a referência da imagem
    label2.pack(side="left", padx=10, pady=10)


def abrir_imagem_matrix_resnet():
    # Cria uma nova janela
    nova_janela = tk.Toplevel(janela)
    nova_janela.title("Imagens")

    # Caminhos das imagens
    caminho_imagem1 = "img//MatrizBiRESNET.png"
    caminho_imagem2 = "img//Matriz6ClassRESNET.png"


    # Carrega as imagens
    img1 = Image.open(caminho_imagem1)
    img1 = ImageTk.PhotoImage(img1.resize((300, 300)))  # Resize the image to desired dimensions
    img2 = Image.open(caminho_imagem2)
    img2 = ImageTk.PhotoImage(img2.resize((300, 300)))  # Resize the image to desired dimensions

    # Cria rótulos para exibir as imagens
    label1 = tk.Label(nova_janela, image=img1)
    label1.image = img1  # Armazena a referência da imagem
    label1.pack(side="left", padx=10, pady=10)

    label2 = tk.Label(nova_janela, image=img2)
    label2.image = img2  # Armazena a referência da imagem
    label2.pack(side="left", padx=10, pady=10)


# Configuração da janela principal
janela = tk.Tk()
janela.title("Papanicolau")
janela.geometry("800x1000")

# Criação do Canvas e Scrollbar
canvas = tk.Canvas(janela)
scrollbar = ttk.Scrollbar(janela, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

# Frames dentro do scrollable_frame
frame_botoes_zoom = ttk.Frame(scrollable_frame)
frame_botoes_histogramas= ttk.Frame(scrollable_frame)
frame_botoes_hu= ttk.Frame(scrollable_frame)

# Variáveis controladoras de estado
imagem_convertida = True
Img_normal = True
Img_conv = False
imagem_cinza = None

# Labels
botao_selecionar = tk.Button(scrollable_frame, text="Selecionar Imagem", command=selecionar_imagem, fg="blue", cursor="hand2")
label_caminho = tk.Label(scrollable_frame, text="")
label_imagem = tk.Label(scrollable_frame)
label_imagem_convertida = tk.Label(scrollable_frame)
label_histograma = tk.Label(scrollable_frame, text="Mostrar histograma de acordo com a imagem atual:")
label_hu = tk.Label(scrollable_frame, text="Mostrar invariantes de hu:")
label_haralick = tk.Label(scrollable_frame, text="Mostrar descritores de haralick:")
label_matriz = tk.Label(scrollable_frame, text="Matriz co-ocorrência descritores de haralick:")
label_SVM = tk.Label(scrollable_frame, text="Classificador Support vector machine (SVM):")
label_RESNET50 = tk.Label(scrollable_frame, text="Classificador RESNET50:")
label_error = tk.Label(scrollable_frame)

# Buttons
botao_tons_cinza = tk.Button(scrollable_frame, text="Exibir imagem em tons de cinza", command=converter_img, height=1, width=30, fg="gray", relief="sunken")
botao_gerar_histograma = tk.Button(frame_botoes_histogramas, text="Gerar", 
                                   command=lambda: histograma.gerar_histograma_hsv(caminho_arquivo=caminho_arquivo) if not Img_conv else histograma.gerar_histograma_tons_cinza(caminho_arquivo=caminho_arquivo), 
                                   height=2, width=15, fg="gray", relief="sunken")
botao_gerar_histograma_16 = tk.Button(frame_botoes_histogramas, text="16 tons", 
                                      command=lambda: histograma.gerar_histograma_16tons(caminho_arquivo=caminho_arquivo) if Img_conv else ..., 
                                      height=2, width=15, fg="gray", relief="sunken")
botao_gerar_histograma_hiv = tk.Button(frame_botoes_histogramas, text="2D HV", 
                                       command=lambda: histograma.gerar_histograma_hiv(caminho_arquivo=caminho_arquivo) if not Img_conv else ...,
                                       height=2, width=15, fg="gray", relief="sunken")
botalharalick = tk.Button(scrollable_frame, text="Mostrar descritores de Haralick", 
                          command=lambda: haralick.calcular_descritores_haralick(caminho_arquivo=caminho_arquivo) if Img_conv else ...,
                          height=2, width=30, fg="gray", relief="sunken")
botaoHu = tk.Button(frame_botoes_hu, text="Mostrar invariantes de HU", 
                    command=lambda: hu.hu_cinza_hsv(caminho_arquivo=caminho_arquivo) if ...else ...,
                      height=2, width=30, fg="gray", relief="sunken")
botao_gerar_mat_coocorrencia = tk.Button(scrollable_frame, text="Gerar matriz de co-ocorrência",
                                        command=lambda: matriz_coocorrencia.gerar_matriz_coocorrencia(caminho_arquivo=caminho_arquivo) if Img_conv else ...,
                                        height=2, width=30, fg="gray", relief="sunken")
botao_gerar_inv_hu_256 = tk.Button(frame_botoes_hu, text="Invariantes de Hu 256", 
                                   command=lambda: hu.gerar_invariantes_hu_256(caminho_arquivo=caminho_arquivo) if ...else ...,
                                   height=2, width=30, fg="gray", relief="sunken")
botao_SVM_Binary_predict = tk.Button(scrollable_frame, text="Classificador binário", 
                                   command=SVMBiPred,
                                   height=2, width=30, fg="gray", relief="sunken")
botao_SVM_6_Class_predict = tk.Button(scrollable_frame, text="Classificador 6 classes", 
                                   command=SVM6ClassPred,
                                   height=2, width=30, fg="gray", relief="sunken")
botao_RESNET50_Binary_predict = tk.Button(scrollable_frame, text="Classificador binário", 
                                   command=RESNET50BiPred,
                                   height=2, width=30, fg="gray", relief="sunken")
botao_RESNET50_6_Class_predict = tk.Button(scrollable_frame, text="Classificador 6 classes", 
                                   command=RESNET506classPred,
                                   height=2, width=30, fg="gray", relief="sunken")
botao_acuracia_svm = tk.Button(scrollable_frame, text="Acuracia_svm", 
                                   command=abrir_imagem_svm,
                                   height=2, width=30, fg="gray", relief="sunken")
botao_acuracia_resnet = tk.Button(scrollable_frame, text="Acuracia_Resnet50", 
                                   command=abrir_imagem_resnet,
                                   height=2, width=30, fg="gray", relief="sunken")
botao_matriz_svm = tk.Button(scrollable_frame, text="Matriz de confusão SVM", 
                                   command=abrir_imagem_matriz_svm,
                                   height=2, width=30, fg="gray", relief="sunken")
botao_matriz_resnet = tk.Button(scrollable_frame, text="Matriz de confusão RESNET50", 
                                   command=abrir_imagem_matrix_resnet,
                                   height=2, width=30, fg="gray", relief="sunken")
# Botões de zoom
botao_zoom_in = tk.Button(frame_botoes_zoom, text="➕", command=zoom_in, height=1, width=5, fg="gray", relief="sunken")
botao_zoom_out = tk.Button(frame_botoes_zoom, text="➖", command=zoom_out, height=1, width=5, fg="gray", relief="sunken")

# Plotar labels na tela
botao_selecionar.pack(pady=3)
botao_tons_cinza.pack()
label_caminho.pack()
frame_botoes_zoom.pack()
botao_zoom_in.pack(side="left")
botao_zoom_out.pack(side="left")
label_imagem.pack()
label_histograma.pack()
frame_botoes_histogramas.pack()
botao_gerar_histograma.pack(padx=6, pady=6, side="left")
botao_gerar_histograma_16.pack(padx=6, pady=6, side="left")
botao_gerar_histograma_hiv.pack(padx=6, pady=6, side="left")
label_hu.pack()
frame_botoes_hu.pack()
botaoHu.pack(padx=6, pady=6, side="left")
botao_gerar_inv_hu_256.pack(padx=6, pady=6, side="left")
label_haralick.pack()
botalharalick.pack(padx=6, pady=6)
label_matriz.pack()
botao_gerar_mat_coocorrencia.pack(padx=6, pady=6)
label_error.pack()
label_SVM.pack()
botao_SVM_Binary_predict.pack()
botao_SVM_6_Class_predict.pack()
botao_acuracia_svm.pack()
botao_matriz_svm.pack()
label_RESNET50.pack()
botao_RESNET50_Binary_predict.pack()
botao_RESNET50_6_Class_predict.pack()
botao_acuracia_resnet.pack()
botao_matriz_resnet.pack()

# Colocar o Canvas e Scrollbar na janela principal
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Executa a aplicação
janela.mainloop()