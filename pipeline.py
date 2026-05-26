"""
Pipeline de leitura automática de placas.

Fluxo:
  1. Lê frames do vídeo (webcam, RTSP ou arquivo)
  2. Detecta movimento (OpenCV - barato)
  3. Detecta veículo (YOLOv8 - local, sem custo de API)
  4. Aplica debounce pra não chamar a API repetidamente
  5. Envia frame pro Plate Recognizer
  6. Faz match contra placas cadastradas (corrige erros de OCR)
  7. Determina entrada ou saída pelo histórico
  8. Grava no SQLite
"""
import os
import time
import sqlite3
from datetime import datetime
from pathlib import Path

import cv2
import requests

import config
from plate_matcher import load_registered_plates, match_plate


# ===============================
# Banco de dados
# ===============================

def init_database():
    conn = sqlite3.connect(config.DATABASE_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            plate_raw TEXT,
            plate_matched TEXT,
            confidence REAL,
            event_type TEXT,
            image_path TEXT
        )
    ''')
    conn.commit()
    return conn


def get_last_event(conn, plate):
    c = conn.cursor()
    c.execute('''
        SELECT event_type, timestamp FROM events
        WHERE plate_matched = ?
        ORDER BY timestamp DESC LIMIT 1
    ''', (plate,))
    return c.fetchone()


def determine_event_type(conn, plate):
    """
    Decide se é entrada ou saída:
      - sem evento anterior → entrada
      - último foi entrada (há > N min) → saída
      - último foi saída (há > N min) → entrada
      - leitura muito recente → ignora (provável duplicação)
    """
    last = get_last_event(conn, plate)
    if last is None:
        return 'entry'

    last_type, last_ts = last
    last_dt = datetime.fromisoformat(last_ts)
    elapsed = (datetime.now() - last_dt).total_seconds()

    if elapsed < config.MIN_GAP_BETWEEN_EVENTS_SECONDS:
        return None  # ignorar leitura duplicada

    return 'exit' if last_type == 'entry' else 'entry'


def save_event(conn, plate_raw, plate_matched, confidence, event_type, image_path):
    c = conn.cursor()
    c.execute('''
        INSERT INTO events (plate_raw, plate_matched, confidence, event_type, image_path)
        VALUES (?, ?, ?, ?, ?)
    ''', (plate_raw, plate_matched, confidence, event_type, image_path))
    conn.commit()


# ===============================
# Plate Recognizer
# ===============================

def send_to_plate_recognizer(image_path):
    try:
        with open(image_path, 'rb') as f:
            res = requests.post(
                'https://api.platerecognizer.com/v1/plate-reader/',
                files={'upload': f},
                data={'regions': config.PLATE_RECOGNIZER_REGION},
                headers={'Authorization': f'Token {config.PLATE_RECOGNIZER_TOKEN}'},
                timeout=15
            )
            res.raise_for_status()
            return res.json()
    except Exception as e:
        print(f"[ERRO] Plate Recognizer: {e}")
        return None


# ===============================
# Pipeline principal
# ===============================

def main():
    # Preparação
    Path(config.IMAGES_DIR).mkdir(exist_ok=True)
    conn = init_database()
    registered = load_registered_plates(config.REGISTERED_PLATES_CSV)
    print(f"[INFO] {len(registered)} placas cadastradas no banco.")

    # YOLO (import aqui pra não atrasar a inicialização caso só queira testar outras partes)
    from ultralytics import YOLO
    print(f"[INFO] Carregando modelo YOLO ({config.YOLO_MODEL})...")
    model = YOLO(config.YOLO_MODEL)

    # Subtrator de fundo (detecção de movimento)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=False
    )

    # Abre vídeo
    print(f"[INFO] Abrindo fonte: {config.VIDEO_SOURCE}")
    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    if not cap.isOpened():
        print("[ERRO] Não consegui abrir a fonte de vídeo.")
        return

    last_api_call = 0
    frame_count = 0

    print("[INFO] Pipeline rodando. Pressione 'q' na janela pra sair.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Fim do stream.")
                break

            frame_count += 1
            if frame_count % config.FRAME_SKIP != 0:
                continue

            # Aplica ROI
            if config.ROI:
                x1, y1, x2, y2 = config.ROI
                work_frame = frame[y1:y2, x1:x2]
            else:
                work_frame = frame

            # 1. Detecção de movimento
            fg_mask = bg_subtractor.apply(work_frame)
            motion_area = cv2.countNonZero(fg_mask)

            display = frame.copy()
            if config.ROI:
                cv2.rectangle(display, (config.ROI[0], config.ROI[1]),
                              (config.ROI[2], config.ROI[3]), (0, 255, 255), 2)

            status = "aguardando"

            if motion_area > config.MIN_MOTION_AREA:
                status = f"movimento ({motion_area}px)"
                now = time.time()

                if now - last_api_call < config.DEBOUNCE_SECONDS:
                    status += " [cooldown]"
                else:
                    # 2. Detecção de veículo
                    results = model(work_frame, conf=config.YOLO_CONFIDENCE, verbose=False)
                    vehicle_detected = False

                    for r in results:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            if cls_id in config.VEHICLE_CLASSES:
                                vehicle_detected = True
                                x1b, y1b, x2b, y2b = map(int, box.xyxy[0])
                                if config.ROI:
                                    x1b += config.ROI[0]; x2b += config.ROI[0]
                                    y1b += config.ROI[1]; y2b += config.ROI[1]
                                cv2.rectangle(display, (x1b, y1b), (x2b, y2b),
                                              (0, 255, 0), 2)

                    if vehicle_detected:
                        status = "veiculo detectado - enviando"
                        last_api_call = now

                        # Salva imagem
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_path = os.path.join(config.IMAGES_DIR, f"capture_{ts}.jpg")
                        cv2.imwrite(image_path, frame)

                        # 3. API
                        response = send_to_plate_recognizer(image_path)

                        if response and response.get('results'):
                            result = response['results'][0]
                            plate_raw = result['plate'].upper()
                            confidence = result['score']

                            print(f"\n[OCR] {plate_raw} ({confidence:.1%})")

                            # 4. Match com banco
                            matched = match_plate(plate_raw, registered)
                            if matched:
                                plate_final, dist, desc = matched
                                tag = f" - {desc}" if desc else ""
                                print(f"[MATCH] {plate_final}{tag}  [dist={dist:.1f}]")
                            else:
                                plate_final = plate_raw
                                print(f"[SEM MATCH] Placa não está cadastrada.")

                            # 5. Entrada/Saída
                            event_type = determine_event_type(conn, plate_final)
                            if event_type is None:
                                print(f"[SKIP] Leitura muito próxima do último evento.")
                            else:
                                save_event(conn, plate_raw, plate_final,
                                           confidence, event_type, image_path)
                                tipo = "ENTRADA" if event_type == 'entry' else "SAÍDA "
                                print(f"[REGISTRO] >>> {tipo} <<< {plate_final}")
                        else:
                            print("[INFO] Nenhuma placa detectada no frame.")
                    else:
                        status = "movimento sem veiculo"

            # Mostra janela
            if config.SHOW_WINDOW:
                cv2.putText(display, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow('Leitor de Placas', display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrompido pelo usuário.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        conn.close()
        print("[INFO] Encerrado.")


if __name__ == '__main__':
    main()
