"""
Configurações do pipeline de leitura de placas.
Edite os valores conforme seu ambiente.
"""

"""
Configurações do pipeline de leitura de placas.

Valores sensíveis (token API, URL do DVR) vêm do arquivo .env.
Edite o .env (copie do .env.example) — NÃO commit o .env no Git.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _get_video_source():
    """
    VIDEO_SOURCE pode ser:
      - número inteiro (webcam): "0", "1"...
      - caminho de arquivo: "video.mp4"
      - URL RTSP: "rtsp://..."
    """
    src = os.getenv("VIDEO_SOURCE", "0")
    # Se for um número, converte pra int (webcam)
    if src.isdigit():
        return int(src)
    return src


# ===============================
# Fonte de vídeo (do .env)
# ===============================
VIDEO_SOURCE = _get_video_source()


# ===============================
# Plate Recognizer (do .env)
# ===============================
PLATE_RECOGNIZER_TOKEN = os.getenv("PLATE_RECOGNIZER_TOKEN", "")
PLATE_RECOGNIZER_REGION = os.getenv("PLATE_RECOGNIZER_REGION", "br")

if not PLATE_RECOGNIZER_TOKEN:
    print("[AVISO] PLATE_RECOGNIZER_TOKEN não configurado no .env")


# ===============================
# Detecção de veículo (YOLO)
# ===============================
YOLO_MODEL = "yolov8n.pt"       # nano = mais rápido. Pode trocar por yolov8s.pt se tiver GPU.
YOLO_CONFIDENCE = 0.5
# Classes COCO consideradas veículo: car(2), motorcycle(3), bus(5), truck(7)
VEHICLE_CLASSES = [2, 3, 5, 7]


# ===============================
# Detecção de movimento
# ===============================
MIN_MOTION_AREA = 5000   # área mínima em pixels pra considerar que algo se moveu


# ===============================
# Região de interesse (ROI)
# ===============================
# Defina como (x1, y1, x2, y2) em pixels. None = imagem inteira.
# Útil pra ignorar áreas onde não passam caminhões (céu, prédios atrás).
ROI = None


# ===============================
# Performance
# ===============================
FRAME_SKIP = 5                # processa 1 a cada N frames (economiza CPU)
DEBOUNCE_SECONDS = 30         # após enviar pra API, aguarda N seg antes de mandar outra


# ===============================
# Entrada / Saída
# ===============================
# Tempo mínimo entre dois eventos da mesma placa pra não duplicar
MIN_GAP_BETWEEN_EVENTS_SECONDS = 10 * 60  # 10 minutos


# ===============================
# Arquivos
# ===============================
REGISTERED_PLATES_CSV = "registered_plates.csv"
DATABASE_PATH = "events.db"
IMAGES_DIR = "captures"
SAVE_IMAGES = True


# ===============================
# Visualização
# ===============================
SHOW_WINDOW = True   # mostra janela com vídeo + bounding boxes. Coloque False em produção.
