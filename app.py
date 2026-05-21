import os
# Força o uso do Keras clássico para compatibilidade total
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
from tf_keras.models import load_model  # Voltamos ao padrão robusto
from PIL import Image, ImageOps
import numpy as np

# --- CONFIGURAÇÃO DA PÁGINA (Aba e Ícone Profissional) ---
st.set_page_config(
    page_title="AI de Gestos - Apresentação Final", 
    page_icon="🤖", 
    layout="wide" 
)
# ... (Mantenha todo o resto do seu código CSS igualzinho abaixo dessa linha) ...

# --- CARREGAR MODELO (Lógica Interna Atualizada) ---
@st.cache_resource
def carregar_modelo_ia():
    # Carrega usando a biblioteca leve
    modelo = load_model("keras_Model.h5", compile=False)
    return modelo

modelo_ia = carregar_modelo_ia()

# ... (Mantenha o restante do código exatamente como estava) ...

# --- CSS CUSTOMIZADO AVANÇADO (Design Unificado e Coeso) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }

    .unificado-container {
        background: #f4f4f4; 
        padding: 30px;
        border-radius: 20px;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.05);
        display: flex;
        gap: 30px;
    }

    .unificado-container .control-panel {
        background-color: #1e3a8a; 
        color: white;
        padding: 25px;
        border-radius: 15px;
        flex: 1; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        border: 1px solid #1a2e6e;
    }
    
    .unificado-container .control-panel .card-title {
        color: white !important;
        font-weight: 700;
        font-size: 1.4rem;
        margin-bottom: 10px;
        border-bottom: 2px solid #5a7bfa;
        padding-bottom: 5px;
        display: inline-block;
    }
    
    .unificado-container .control-panel p {
        color: rgba(255,255,255,0.8);
    }

    div.stFileUploader > label {
        display: none;
    }
    div.stFileUploader > section {
        border: 2px dashed #5a7bfa;
        border-radius: 12px;
        padding: 25px;
        background-color: #2446a6;
        transition: background-color 0.3s;
    }
    div.stFileUploader > section:hover {
        background-color: #1c357d;
    }

    .unificado-container .analysis-panel {
        background-color: #1e3a8a; 
        color: white;
        padding: 25px;
        border-radius: 15px;
        flex: 1.2; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        border: 1px solid #1a2e6e;
    }

    .unificado-container .analysis-panel .card-title {
        color: white !important;
        font-weight: 700;
        font-size: 1.4rem;
        margin-bottom: 10px;
        border-bottom: 2px solid #5a7bfa;
        padding-bottom: 5px;
        display: inline-block;
    }

    .analysis-panel .result-section {
        background-color: rgba(255,255,255,0.05);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
    }
    
    .analysis-panel .result-text {
        font-size: 3rem;
        font-weight: 800;
        color: #a7f3d0; 
        margin-top: 10px;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 0 2px 5px rgba(0,0,0,0.5);
    }
    
    .low-confidence {
        color: #fbd38d; 
    }

    .confidence-section {
        margin-top: 15px;
    }
    .progress-label {
        font-weight: 600;
        color: white;
        margin-bottom: 5px;
        opacity: 0.9;
    }
    .progress-bg {
        background-color: rgba(255,255,255,0.1);
        border-radius: 10px;
        width: 100%;
        height: 18px;
        overflow: hidden;
    }
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
    }

    .event-footer {
        text-align: center;
        padding: 20px;
        margin-top: 30px;
        border-top: 1px solid #e0e0e0;
        color: #777;
    }
</style>
""", unsafe_allow_html=True)

# --- CARREGAR MODELO (Lógica Interna) ---
@st.cache_resource
def carregar_modelo_ia():
    modelo = load_model("keras_Model.h5", compile=False)
    return modelo

modelo_ia = carregar_modelo_ia()

# Carregar os rótulos (labels.txt)
with open("labels.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip().split(" ", 1)[-1] for line in f.readlines()]

# --- LAYOUT UNIFICADO DO EVENTO ---
st.markdown('<div class="unificado-container">', unsafe_allow_html=True)

    # --- PAINEL DE CONTROLE (Esquerda) ---
st.markdown('<div class="control-panel">', unsafe_allow_html=True)
st.markdown('<p class="card-title">Apresentação: IA Gestos</p>', unsafe_allow_html=True)
st.markdown('<p>Conclusão de Desafio - Apresentação Final</p>', unsafe_allow_html=True)
st.markdown('<p class="progress-label">Suba uma imagem nítida para análise:</p>', unsafe_allow_html=True)
    
arquivo_imagem = st.file_uploader("", type=["jpg", "png", "jpeg"])
st.markdown('</div>', unsafe_allow_html=True) 

    # --- PAINEL DE ANÁLISE (Direita) ---
st.markdown('<div class="analysis-panel">', unsafe_allow_html=True)
st.markdown('<p class="card-title">Detecção de Gesto:</p>', unsafe_allow_html=True)

if arquivo_imagem is not None:
    image = Image.open(arquivo_imagem).convert("RGB")
    
    # Mostra a foto enviada pelo usuário
    st.image(image, caption='Sua foto enviada', use_column_width=True)
    
    # Processamento e preparação da imagem
    size = (224, 224)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image_resized)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Executa a inteligência artificial
    prediction = modelo_ia.predict(data)
    index = np.argmax(prediction)
    nome_classe = class_names[index]
    confianca = prediction[0][index]
    porcentagem = confianca * 100

    # Estrutura visual do resultado dentro do painel escuro
    st.markdown('<div class="result-section">', unsafe_allow_html=True)
    st.write("Gesto Identificado:")

    if porcentagem >= 70:
        st.markdown(f'<p class="result-text">{nome_classe.upper()}</p>', unsafe_allow_html=True)
        color_bar = "#6ee7b7" # Verde suave do design para alta certeza
    else:
        st.markdown(f'<p class="result-text low-confidence">{nome_classe.upper()}</p>', unsafe_allow_html=True)
        color_bar = "#fbd38d" # Laranja para baixa certeza
        st.warning("Nota: Confiança abaixo do ideal. Considere refazer a foto.")

    # Seção da Barra de Progresso Customizada
    st.markdown('<div class="confidence-section">', unsafe_allow_html=True)
    st.markdown(f'<p class="progress-label">Taxa de Certeza: {porcentagem:.1f}%</p>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="progress-bg">
            <div class="progress-bar" style="width: {porcentagem}%; background-color: {color_bar};"></div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True) 
    st.markdown('</div>', unsafe_allow_html=True) 

else:
    # Caso nenhuma imagem tenha sido enviada ainda
    st.warning("Aguardando o upload de um arquivo para iniciar a análise da IA.")

st.markdown('</div>', unsafe_allow_html=True) # Fecha a div da análise
st.markdown('</div>', unsafe_allow_html=True) # Fecha a div do container unificado

# Rodapé institucional do Desafio
st.markdown("""
<div class="event-footer">
    <p>Desafio Concluido com Sucesso - Projeto de Apresentação Final - IA Gestos 2024</p>
</div>
""", unsafe_allow_html=True)