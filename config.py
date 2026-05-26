"""
Configurações do pipeline de leitura de placas.
Edite os valores conforme seu ambiente.
"""

# ===============================
# Fonte de vídeo
# ===============================
# Opções:
#   - 0 (ou 1, 2...): webcam local — bom pra teste inicial
#   - "caminho/do/video.mp4": arquivo de vídeo gravado
#   - "rtsp://usuario:senha@IP:554/cam/realmonitor?channel=1&subtype=0": stream do DVR Intelbras
VIDEO_SOURCE = 0


# ===============================
# Plate Recognizer (API)
# ===============================
PLATE_RECOGNIZER_TOKEN = "COLE_SEU_TOKEN_AQUI"
PLATE_RECOGNIZER_REGION = "br"


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
