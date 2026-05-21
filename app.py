import os

# =========================================================
# CONFIGURAÇÃO KERAS
# =========================================================

os.environ["TF_USE_LEGACY_KERAS"] = "1"

# =========================================================
# IMPORTS
# =========================================================

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tf_keras.models import load_model
from tf_keras.layers import DepthwiseConv2D

# =========================================================
# CONFIG DA PÁGINA
# =========================================================

st.set_page_config(
    page_title="Vision Gesture AI",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# CSS NOVO FRONT-END
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0f172a;
}

/* REMOVE PADDING PADRÃO */

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    max-width: 1400px;
}

/* FUNDO PRINCIPAL */

.main-background {
    background:
        radial-gradient(circle at top left, #312e81 0%, transparent 35%),
        radial-gradient(circle at bottom right, #0f766e 0%, transparent 35%),
        #020617;

    border-radius: 28px;
    padding: 40px;
    min-height: auto;
}

/* HERO */

.hero-section {
    margin-bottom: 40px;
}

.hero-title {
    font-size: 4rem;
    font-weight: 900;
    color: white;
    line-height: 1;
    margin-bottom: 10px;
    letter-spacing: -2px;
}

.hero-gradient {
    background: linear-gradient(90deg,#60a5fa,#22d3ee,#34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.1rem;
    max-width: 700px;
}

/* GRID */

.custom-grid {
    display: grid;
    grid-template-columns: 420px 1fr;
    gap: 25px;
}

/* CARDS */

.glass-card {
    background: rgba(15,23,42,0.75);
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 28px;
    box-shadow:
        0 10px 40px rgba(0,0,0,0.35),
        inset 0 1px 0 rgba(255,255,255,0.03);
}

/* TITULOS */

.section-label {
    color: #38bdf8;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.section-title {
    color: white;
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 12px;
}

.section-text {
    color: #94a3b8;
    line-height: 1.7;
    margin-bottom: 25px;
}

/* UPLOADER */

div.stFileUploader > label {
    display: none;
}

div.stFileUploader > section {
    border: 2px dashed rgba(56,189,248,0.4);
    border-radius: 20px;
    padding: 40px 20px;
    background: rgba(15,23,42,0.5);
    transition: 0.3s;
}

div.stFileUploader > section:hover {
    border-color: #22d3ee;
    background: rgba(30,41,59,0.7);
}

/* RESULTADO */

.result-container {
    margin-top: 25px;
    padding: 30px;
    border-radius: 22px;
    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.04),
            rgba(255,255,255,0.01)
        );
    border: 1px solid rgba(255,255,255,0.05);
}

.result-title {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.result-gesture {
    font-size: 4rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 20px;
    letter-spacing: -2px;
}

.high-result {
    color: #34d399;
}

.low-result {
    color: #fb923c;
}

/* BARRA */

.confidence-wrapper {
    margin-top: 20px;
}

.confidence-text {
    color: white;
    font-weight: 600;
    margin-bottom: 10px;
}

.progress-bg {
    width: 100%;
    height: 18px;
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.5s ease;
}

/* STATUS */

.status-good {
    margin-top: 18px;
    color: #86efac;
    font-weight: 600;
}

.status-warning {
    margin-top: 18px;
    color: #fdba74;
    font-weight: 600;
}

/* FOOTER */

.footer {
    text-align: center;
    margin-top: 35px;
    color: #64748b;
    font-size: 0.9rem;
}
@media (max-width: 768px) {
    .hero-title {
        font-size: 2.5rem !important; /* Título menor no celular */
        letter-spacing: -1px !important;
    }
    
    .main-background {
        padding: 20px !important; /* Menos margem interna no celular */
    }
    
    .custom-grid {
        grid-template-columns: 1fr !important; /* Colunas viram uma linha única */
    }
    
    .result-gesture {
        font-size: 2.8rem !important; /* Nome do gesto mais compacto */
    }
}            


</style>
""", unsafe_allow_html=True)

# =========================================================
# CLASSE DE COMPATIBILIDADE (DEPTHWISE)
# =========================================================

class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(**kwargs)

# =========================================================
# CARREGAR MODELO (RESOLVE BUG DE MULTIPLOS TENSORES)
# =========================================================

@st.cache_resource
def carregar_modelo():
    try:
        # Carrega o modelo aplicando o remendo do DepthwiseConv2D
        modelo_bruto = load_model(
            "keras_model.h5", 
            compile=False,
            custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}
        )
        
        # Reconstrói a saída usando API funcional pura para matar o erro do Sequential.call()
        inputs = modelo_bruto.inputs
        if isinstance(inputs, list):
            outputs = modelo_bruto(inputs[0])
        else:
            outputs = modelo_bruto(inputs)
            
        modelo_corrigido = tf.keras.Model(inputs=inputs, outputs=outputs)
        return modelo_corrigido
    except Exception as e:
        st.error(f"Erro ao inicializar modelo de visão computacional: {e}")
        st.stop()

modelo_ia = carregar_modelo()

# =========================================================
# LABELS
# =========================================================

try:
    with open("labels.txt", "r", encoding="utf-8") as f:
        class_names = [
            line.strip().split(" ", 1)[-1]
            for line in f.readlines()
        ]
except Exception as e:
    st.error(f"Erro ao carregar arquivo de rótulos (labels.txt): {e}")
    st.stop()

# =========================================================
# HERO
# =========================================================

st.markdown('<div class="main-background">', unsafe_allow_html=True)

st.markdown("""
<div class="hero-section">

<div class="section-label">
ARTIFICIAL INTELLIGENCE
</div>

<div class="hero-title">
Gesture Recognition<br>
<span class="hero-gradient">Powered by Vision AI</span>
</div>

<div class="hero-subtitle">
Sistema inteligente de reconhecimento visual utilizando
TensorFlow + Streamlit para análise de gestos em tempo real.
</div>

</div>
""", unsafe_allow_html=True)

# =========================================================
# GRID
# =========================================================

col1, col2 = st.columns([1, 1.4], gap="large")

# =========================================================
# COLUNA ESQUERDA
# =========================================================

with col1:

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-label">
    Upload
    </div>

    <div class="section-title">
    Envie uma imagem
    </div>

    <div class="section-text">
    Faça upload de uma foto contendo um gesto para
    a IA analisar e identificar automaticamente.
    </div>
    """, unsafe_allow_html=True)

    arquivo_imagem = st.file_uploader(
        "Upload",
        type=["jpg", "jpeg", "png"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# COLUNA DIREITA
# =========================================================

with col2:

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-label">
    Resultado
    </div>

    <div class="section-title">
    Análise da Inteligência Artificial
    </div>
    """, unsafe_allow_html=True)

    if arquivo_imagem is not None:
        try:
            # Controle dos balões via session_state para não disparar em loop infinito
            if "ultima_img" not in st.session_state or st.session_state.ultima_img != arquivo_imagem.name:
                st.session_state.ultima_img = arquivo_imagem.name
                st.session_state.baloes_ativos = False

            image = Image.open(arquivo_imagem).convert("RGB")

            st.image(
                image,
                use_container_width=True
            )

            # =====================================================
            # PREPROCESSAMENTO
            # =====================================================

            size = (224, 224)

            data = np.ndarray(
                shape=(1, 224, 224, 3),
                dtype=np.float32
            )

            image_resized = ImageOps.fit(
                image,
                size,
                Image.Resampling.LANCZOS
            )

            image_array = np.asarray(image_resized)

            normalized_image_array = (
                image_array.astype(np.float32) / 127.5
            ) - 1

            data[0] = normalized_image_array

            # =====================================================
            # IA
            # =====================================================

            with st.spinner("Analisando gesto..."):
                prediction = modelo_ia.predict(data)

            index = np.argmax(prediction)
            nome_classe = class_names[index]
            confianca = prediction[0][index]
            porcentagem = confianca * 100

            # =====================================================
            # RESULTADO VISUAL DINÂMICO (CORRIGIDO)
            # =====================================================

            if porcentagem >= 70:
                result_class = "high-result"
                bar_color = "linear-gradient(90deg, #34d399, #10b981)"
                status = '<div class="status-good">✓ Reconhecimento realizado com alta confiança</div>'
                if not st.session_state.baloes_ativos:
                    st.balloons()
                    st.session_state.baloes_ativos = True
            else:
                result_class = "low-result"
                bar_color = "linear-gradient(90deg, #fb923c, #f97316)"
                status = '<div class="status-warning">⚠ A confiança ficou abaixo do ideal. Tente melhorar o enquadramento.</div>'

            # HTML linearizado em variável única para impedir que o parser do Streamlit confunda quebras de linha com markdown de código
            html_resultado = f"""
            <div class="result-container">
                <div class="result-title">Gesto Detectado</div>
                <div class="result-gesture {result_class}">{nome_classe.upper()}</div>
                <div class="confidence-wrapper">
                    <div class="confidence-text">Precisão: {porcentagem:.2f}%</div>
                    <div class="progress-bg">
                        <div class="progress-fill" style="width:{porcentagem}%; background:{bar_color};"></div>
                    </div>
                    {status}
                </div>
            </div>
            """

            st.markdown(html_resultado, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Erro ao processar imagem: {e}")

    else:
        st.info(
            "Envie uma imagem para iniciar a análise da IA."
        )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
TensorFlow • Streamlit • Vision AI
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)