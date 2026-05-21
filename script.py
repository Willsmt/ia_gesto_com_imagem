import os
# Força o TensorFlow a usar o ecossistema clássico do Keras 2
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from tf_keras.models import load_model  
from PIL import Image, ImageOps  
import numpy as np

# --- CONFIGURAÇÕES DO USUÁRIO ---
ARQUIVO_IMAGEM = "./teste2.webp" 
LIMIAR_CONFIANCA = 0.70
# ---------------------------------

np.set_printoptions(suppress=True)

print("\n--- Iniciando Sistema de Reconhecimento de Gestos ---")
print("Carregando modelo de IA...")

model = load_model("keras_Model.h5", compile=False)

with open("labels.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip().split(" ", 1)[-1] for line in f.readlines()]

print("Modelo carregado com sucesso!")
print(f"Analisando a imagem: {ARQUIVO_IMAGEM}...\n")

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

try:
    image = Image.open(ARQUIVO_IMAGEM).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # EXECUTA A PREVISÃO
    prediction = model.predict(data)
    index = np.argmax(prediction)
    nome_classe = class_names[index]
    confianca = prediction[0][index]

    # --- MENSAGEM AMIGÁVEL SEM EMOJIS (EVITA O ERRO DE UNICODE) ---
    print("-" * 40)
    print("RESULTADO DA ANÁLISE:")
    print("-" * 40)
    
    if confianca >= LIMIAR_CONFIANCA:
        porcentagem = confianca * 100
        print(f"Com {porcentagem:.1f}% de certeza, eu vi o gesto:")
        print(f"[{nome_classe.upper()}]")
    else:
        print("Humm... Eu olhei para a imagem, mas nao tenho certeza.")
        print(f"A minha maior aposta foi '{nome_classe}', mas com apenas {confianca*100:.1f}% de precisao.")
        print("Tente tirar outra foto mais nitida ou com fundo mais simples!")
    print("-" * 40 + "\n")

except FileNotFoundError:
    print(f"Erro: Nao consegui encontrar o arquivo de imagem '{ARQUIVO_IMAGEM}'.")
except Exception as e:
    print(f"Ocorreu um erro inesperado no processamento.")