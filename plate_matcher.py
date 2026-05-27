"""
Comparação de placas com tolerância a erros comuns de OCR.

Estratégias:
  1. Grupos de confusão de caracteres (8↔B, 9↔0, etc.)
  2. Match contra TODOS os candidatos retornados pelo Plate Recognizer,
     não só o "best". A leitura correta frequentemente aparece como
     segunda ou terceira opção com confiança um pouco menor.
"""
import csv
from typing import Optional, Tuple, List, Dict


# Grupos de caracteres comumente confundidos pelo OCR.
# Grupos podem se sobrepor — a checagem retorna True se ambos os chars
# aparecem em qualquer grupo.
CONFUSION_GROUPS = [
    # Letras ↔ dígitos parecidos
    {'0', 'O', 'D', 'Q'},
    {'1', 'I', 'L'},
    {'2', 'Z'},
    {'5', 'S'},
    {'6', 'G'},
    {'8', 'B'},
    {'7', 'T', 'Y'},
    {'4', 'A'},
    {'9', 'P'},

    # Dígitos entre si (formas curvas similares)
    {'0', '9'},          # ambos arredondados
    {'0', '8'},          # ambos fechados/curvos
    {'6', '8'},          # curvas similares
    {'6', '9'},          # rotação espelhada
    {'8', '9'},          # loops similares
    {'8', '3'},          # metade do 8 parece 3
    {'3', '5'},          # topo curvo similar
    {'5', '6'},          # ambos com curva inferior
    {'2', '7'},          # ângulo superior similar

    # Letras entre si
    {'M', 'N', 'H'},
    {'C', 'G'},
    {'V', 'Y', 'U'},
    {'K', 'X'},
    {'R', 'P', 'B'},
]


def _normalize(plate: str) -> str:
    """Remove hífen, espaço e padroniza maiúsculas."""
    return plate.upper().replace('-', '').replace(' ', '').strip()


def chars_confusable(a: str, b: str) -> bool:
    """True se dois caracteres aparecem juntos em algum grupo de confusão."""
    if a == b:
        return True
    for group in CONFUSION_GROUPS:
        if a in group and b in group:
            return True
    return False


def weighted_distance(a: str, b: str) -> float:
    """
    Distância customizada:
      - iguais: 0
      - confundíveis: 0.3
      - diferentes: 1.0
      - tamanhos diferentes: 1.0 por caractere extra
    """
    a = _normalize(a)
    b = _normalize(b)

    if len(a) != len(b):
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
    """Carrega placas do CSV. Espera coluna 'plate' (e 'description' opcional)."""
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
    Match simples: uma única leitura OCR contra o banco.
    Retorna (placa_correta, distância, descrição) ou None.
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


def match_plate_with_candidates(
    candidates: List[Dict],
    registered: List[Dict[str, str]],
    max_distance: float = 1.5
) -> Optional[Dict]:
    """
    Match avançado: testa TODOS os candidatos do OCR contra o banco e
    escolhe a combinação com melhor score (combinando confiança do OCR
    e proximidade do match).

    `candidates` no formato do Plate Recognizer: [{'plate': 'xxx', 'score': 0.85}, ...]

    Retorna dict com:
      - matched_plate: placa cadastrada que matchou
      - distance: distância da edição
      - description: descrição da placa cadastrada
      - via_candidate: qual candidato do OCR levou ao match
      - candidate_score: confiança do OCR daquele candidato
    Ou None se nada matchar.
    """
    if not registered or not candidates:
        return None

    best = None
    best_score = -1.0

    for cand in candidates:
        cand_plate = _normalize(cand.get('plate', ''))
        cand_score = cand.get('score', 0)
        if not cand_plate:
            continue

        for entry in registered:
            dist = weighted_distance(cand_plate, entry['plate'])
            if dist > max_distance:
                continue

            # Score combinado: confiança OCR alta + distância baixa = melhor
            # Normaliza distância pra 0-1 e combina com score
            dist_factor = 1 - (dist / (max_distance + 1))
            combined = cand_score * 0.6 + dist_factor * 0.4

            if combined > best_score:
                best_score = combined
                best = {
                    'matched_plate': entry['plate'],
                    'distance': dist,
                    'description': entry['description'],
                    'via_candidate': cand_plate,
                    'candidate_score': cand_score,
                }

    return best