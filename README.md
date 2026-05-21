

---

# Vision Gesture AI 🧠

Este projeto é uma aplicação de Inteligência Artificial para reconhecimento de gestos em tempo real, construída com **TensorFlow** e **Streamlit**. O sistema processa imagens enviadas pelo usuário, aplica pré-processamento de dados e utiliza um modelo de visão computacional para classificar gestos com alta precisão.

## 🚀 Funcionalidades

* **Reconhecimento de Gestos:** Identificação automática baseada em modelos treinados.
* **Interface Moderna:** Design responsivo estilo *glassmorphism* com tema *dark*.
* **Feedback Visual:** Barra de progresso dinâmica e indicadores de confiança.
* **Pronto para Deploy:** Configurado via Docker para fácil implementação em nuvem.

## 🛠 Tecnologias Utilizadas

* **Python 3.12**
* **TensorFlow / Keras:** Motor de inferência da IA.
* **Streamlit:** Framework para criação da interface web.
* **Pillow (PIL):** Processamento e manipulação de imagens.
* **Docker:** Containerização para portabilidade e deploy.

## 📋 Pré-requisitos

Para rodar este projeto localmente, você precisará de:

* Python 3.12+
* Docker instalado (opcional, para ambiente isolado)
* O arquivo do seu modelo: `keras_model.h5`
* O arquivo de rótulos: `labels.txt`

## ⚙️ Como Instalar e Rodar

1. **Clone o repositório:**
```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio

```


2. **Instale as dependências:**
```bash
pip install -r requirements.txt

```


3. **Execute o projeto:**
```bash
streamlit run app.py

```





```


```

## 🧠 Arquitetura do Modelo

O sistema utiliza uma rede neural convolucional para extrair características visuais. O pré-processamento garante que cada imagem seja redimensionada para 224x224 pixels e normalizada, conforme exigido pelo modelo treinado.

![Arquitetura](assets/modelo.demo.jpg)

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir *issues* ou enviar *pull requests*.

## ✒️ Autor

* **Willians Martins** 

---

