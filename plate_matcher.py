"""
Comparação de placas com tolerância a erros comuns de OCR.

Faz match da placa lida pelo OCR contra o banco de placas cadastradas,
considerando que caracteres como 8/B, 0/O/D, 1/I etc. são frequentemente
confundidos.
"""
import csv
from typing import Optional, Tuple, List, Dict


# Grupos de caracteres que o OCR comumente confunde entre si.
# Substituição dentro do mesmo grupo é considerada "barata".
CONFUSION_GROUPS = [
    {'0', 'O', 'D', 'Q'},
    {'1', 'I', 'L'},
    {'2', 'Z'},
    {'5', 'S'},
    {'6', 'G', '8'},     # 6 e 8 são frequentemente confundidos em câmeras de baixa resolução
    {'8', 'B', '6'},
    {'7', 'T', 'Y'},
    {'4', 'A'},
    {'9', 'P'},
    {'M', 'N', 'H'},
    {'C', 'G'},
    {'V', 'Y'},
    {'U', 'V'},
]


def _normalize(plate: str) -> str:
    """Remove hífen, espaço e padroniza maiúsculas."""
    return plate.upper().replace('-', '').replace(' ', '').strip()


def chars_confusable(a: str, b: str) -> bool:
    """Retorna True se dois caracteres podem ser confundidos pelo OCR."""
    if a == b:
        return True
    for group in CONFUSION_GROUPS:
        if a in group and b in group:
            return True
    return False


def weighted_distance(a: str, b: str) -> float:
    """
    Distância customizada:
      - caracteres iguais: 0
      - confundíveis (mesmo grupo): 0.3
      - diferentes: 1.0
      - diferença de tamanho: 1.0 por caractere
    """
    a = _normalize(a)
    b = _normalize(b)

    if len(a) != len(b):
        # Tamanhos diferentes: comparação inviável para placas (sempre 7 chars no BR).
        return abs(len(a) - len(b)) + min(len(a), len(b))

    cost = 0.0
    for ca, cb in zip(a, b):
        if ca == cb:
            continue
        elif chars_confusable(ca, cb):
            cost += 0.3
        else:
            cost += 1.0
    return cost


def load_registered_plates(csv_path: str) -> List[Dict[str, str]]:
    """
    Carrega placas cadastradas do CSV.
    Espera colunas: 'plate' (obrigatória), 'description' (opcional).
    """
    plates = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                plate = _normalize(row.get('plate', ''))
                if plate:
                    plates.append({
                        'plate': plate,
                        'description': row.get('description', '').strip()
                    })
    except FileNotFoundError:
        print(f"[AVISO] {csv_path} não encontrado. Match contra banco desabilitado.")
    return plates


def match_plate(
    ocr_plate: str,
    registered: List[Dict[str, str]],
    max_distance: float = 1.5
) -> Optional[Tuple[str, float, str]]:
    """
    Encontra a placa cadastrada mais próxima da leitura do OCR.

    Retorna (placa_correta, distância, descrição) ou None se nada matchar
    dentro da tolerância.
    """
    if not registered:
        return None

    ocr_clean = _normalize(ocr_plate)
    best = None
    best_dist = float('inf')

    for entry in registered:
        dist = weighted_distance(ocr_clean, entry['plate'])
        if dist < best_dist:
            best_dist = dist
            best = entry

    if best and best_dist <= max_distance:
        return (best['plate'], best_dist, best['description'])
    return None
