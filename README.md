# Pipeline de Leitura de Placas

Leitura automática de placas via stream de vídeo (webcam, arquivo ou RTSP do DVR Intelbras), com filtros locais pra economizar chamadas à API do Plate Recognizer.

## Estrutura do funil

```
Frame de vídeo
    ↓
[Filtro 1] Detecção de movimento  (OpenCV, custo zero)
    ↓
[Filtro 2] Detecção de veículo    (YOLOv8 local, custo baixo)
    ↓
[Filtro 3] Debounce/cooldown      (evita repetição)
    ↓
[API] Plate Recognizer            (chamada paga)
    ↓
[Match] Fuzzy match no banco      (corrige erros de OCR)
    ↓
[Lógica] Entrada ou Saída?        (baseado no histórico)
    ↓
SQLite
```

## Instalação

Recomendado usar virtualenv:

```bash
python3 -m venv venv
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

Na primeira execução, o YOLO baixa o modelo automaticamente (~6 MB pro `yolov8n.pt`).

## Configuração

Edite `config.py`:

1. **`PLATE_RECOGNIZER_TOKEN`** — cole sua chave da API
2. **`VIDEO_SOURCE`** — defina a fonte:
   - `0` pra webcam (teste rápido)
   - `"video.mp4"` pra arquivo
   - `"rtsp://usuario:senha@IP:554/cam/realmonitor?channel=1&subtype=0"` pro DVR Intelbras
3. **`ROI`** (opcional) — `(x1, y1, x2, y2)` pra limitar a análise a uma região
4. **`DEBOUNCE_SECONDS`** — tempo de cooldown entre chamadas à API
5. **`MIN_GAP_BETWEEN_EVENTS_SECONDS`** — tempo mínimo entre dois eventos da mesma placa (default 10min)

Edite `registered_plates.csv` com as placas dos seus caminhões. Formato:

```csv
plate,description
ABC1D23,Caminhão João - Mercedes
```

## Execução

```bash
python pipeline.py
```

Pra ver os eventos registrados:

```bash
python view_events.py              # últimos 50
python view_events.py ABC1D23      # histórico de uma placa
```

## Como ajustar pra economizar API

- **Aumentar `DEBOUNCE_SECONDS`**: se o tempo médio entre caminhões é maior que 30s, aumente. Garante 1 leitura por caminhão.
- **Apertar a ROI**: se o caminhão só ocupa metade da imagem, limite a análise àquela região. Diminui falsos positivos e processamento.
- **Aumentar `MIN_MOTION_AREA`**: se há movimento pequeno (galhos, sombras), suba esse valor pra ignorar.
- **`FRAME_SKIP` maior**: processa menos frames por segundo. Em portaria onde caminhão fica visível 5+ segundos, pode subir tranquilo.

## Estimativa de uso da API

Com debounce de 30s e detecção bem ajustada: **1 chamada por caminhão**.

- 30 caminhões/dia × 2 (entrada + saída) = 60 chamadas/dia
- Mês: ~1.800 chamadas → cabe no tier gratuito (2.500/mês)

## Detalhes do match fuzzy

O `plate_matcher.py` reconhece grupos de caracteres comumente confundidos:

| Confusão típica |
|---|
| 0 ↔ O ↔ D ↔ Q |
| 1 ↔ I ↔ L |
| 8 ↔ B |
| 5 ↔ S |
| 6 ↔ G |
| 2 ↔ Z |
| 7 ↔ T ↔ Y |

Substituição dentro do mesmo grupo custa 0.3; substituição "real" custa 1.0. Por padrão aceita match até distância 1.5 — o que cobre 2 erros de OCR confundíveis sem aceitar placas realmente diferentes.

## Próximos passos sugeridos

- **Web UI**: dashboard pra ver entradas/saídas em tempo real (Flask/FastAPI + SQLite)
- **Notificações**: webhook ou e-mail quando uma placa específica entra
- **Múltiplas câmeras**: rodar uma instância do `pipeline.py` por câmera, todas gravando no mesmo banco
- **Migrar pra Plate Recognizer local**: trocar a URL da API pelo container Docker local quando o volume crescer
