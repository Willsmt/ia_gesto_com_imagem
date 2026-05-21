import os
# Força o uso do Keras clássico para compatibilidade
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# --- CONFIGURAÇÃO DA PÁGINA (Aba e Ícone) ---
st.set_page_config(
    page_title="IA de Gestos - Conclusão de Desafio", 
    page_icon="🤖", 
    layout="wide" # Usa a largura total da tela
)

# --- CSS CUSTOMIZADO (O "pulo do gato" para o design) ---
st.markdown("""
<style>
    /* Importando Fonte Moderna */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    /* Aplica fonte globalmente */
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    /* Remove padding padrão do Streamlit no topo */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }

    /* Estilo do Banner de Cabeçalho */
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .hero-banner h1 {
        font-weight: 700;
        margin-bottom: 5px;
        color: white !important;
    }
    .hero-banner p {
        font-weight: 300;
        opacity: 0.9;
    }

    /* Estilo dos Cartões (Cards) */
    .custom-card {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border: 1px solid #f0f0f0;
    }
    .card-title {
        color: #4a148c;
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 15px;
        border-bottom: 2px solid #764ba2;
        padding-bottom: 5px;
        display: inline-block;
    }

    /* Customização do Botão de Upload */
    div.stFileUploader > label {
        display: none; /* Esconde a label padrão */
    }
    div.stFileUploader > section {
        border: 2px dashed #764ba2;
        border-radius: 10px;
        padding: 20px;
        background-color: #f9f9ff;
        transition: background-color 0.3s;
    }
    div.stFileUploader > section:hover {
        background-color: #f0f0f5;
    }

    /* Estilo do Texto de Resultado */
    .result-text {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2e7d32; /* Verde */
        text-align: center;
        margin-top: 10px;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .low-confidence {
        color: #f57c00; /* Laranja */
    }

    /* Estilo da Barra de Progresso Customizada */
    .progress-label {
        font-weight: 600;
        color: #555;
        margin-bottom: 5px;
    }
    .progress-bg {
        background-color: #e0e0e0;
        border-radius: 10px;
        width: 100%;
        height: 15px;
        overflow: hidden;
    }
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# --- CABEÇALHO (FRONT-END CUSTOMIZADO) ---
st.markdown("""
<div class="hero-banner">
    <h1>Apresentação Final: IA de Gestos</h1>
    <p>Desafio de Conclusão - Reconhecimento de Sinais em Tempo Real</p>
</div>
""", unsafe_allow_html=True)

# --- CARREGAR MODELO ---
@st.cache_resource
def carregar_modelo_ia():
    with st.spinner('Carregando cérebro da IA...'):
        modelo = load_model("keras_Model.h5", compile=False)
        return modelo

modelo_ia = carregar_modelo_ia()

# Carregar os rótulos (labels.txt)
with open("labels.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip().split(" ", 1)[-1] for line in f.readlines()]

# --- LAYOUT PRINCIPAL (Duas Colunas) ---
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    # --- CARTÃO DE UPLOAD (FRONT-END) ---
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">1. Envie sua Imagem</p>', unsafe_allow_html=True)
    st.write("Suba uma foto nítida e bem iluminada do seu gesto.")
    
    # Widget de Upload
    arquivo_imagem = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    # Fecha o cartão
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # --- CARTÃO DE RESULTADO (FRONT-END) ---
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">2. Análise da Inteligência Artificial</p>', unsafe_allow_html=True)
    st.write("")

    if arquivo_imagem is not None:
        image = Image.open(arquivo_imagem).convert("RGB")
        
        # Mostra a imagem enviada (centralizada no card)
        st.image(image, caption='Sua foto', )
        
        with st.spinner('A IA está analisando...'):
            # Prepara a imagem
            size = (224, 224)
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            image_array = np.asarray(image_resized)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data[0] = normalized_image_array

            # Executa a previsão
            prediction = modelo_ia.predict(data)
            index = np.argmax(prediction)
            nome_classe = class_names[index]
            confianca = prediction[0][index]
            porcentagem = confianca * 100

        st.markdown("---")
        st.write("Deteção:")

        # --- EXIBIÇÃO VISUAL DO RESULTADO ---
        if porcentagem >= 70:
            st.markdown(f'<p class="result-text">{nome_classe.upper()}</p>', unsafe_allow_html=True)
            color_bar = "#2e7d32" # Verde
            st.balloons()
        else:
            st.markdown(f'<p class="result-text low-confidence">{nome_classe.upper()}</p>', unsafe_allow_html=True)
            color_bar = "#f57c00" # Laranja
            st.info("Nota: Confiança baixa. Tente melhorar a foto.")

        # Barra de Progresso Customizada (Confiança)
        st.markdown(f'<p class="progress-label">Certeza: {porcentagem:.1f}%</p>', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="progress-bg">
                <div class="progress-bar" style="width: {porcentagem}%; background-color: {color_bar};"></div>
            </div>
        """, unsafe_allow_html=True)

    else:
        # Mensagem placeholder quando não há imagem
        st.warning("Aguardando o upload da imagem para iniciar a análise.")

    # Fecha o cartão
    st.markdown('</div>', unsafe_allow_html=True)

# Footer simples
st.markdown("---")
st.markdown("<p style='text-align: center; color: #777;'>Desafio Concluído com Sucesso - IA de Gestos 2024</p>", unsafe_allow_html=True)