FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libgtk2.0-dev \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir \
    opencv-python-headless==4.8.1.78 \
    ultralytics==8.0.230 \
    numpy==1.24.3 \
    streamlit==1.29.0 \
    pillow==10.1.0

COPY . .

ENV DISPLAY=:0

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
