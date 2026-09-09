FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Download the AI model during Docker build
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/ayush2635/Dhwani-Multilingual-Deepfake-Audio-Detection-Model/resolve/main/best_model.onnx', '/app/best_model.onnx')"

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
