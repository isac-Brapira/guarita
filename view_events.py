"""
Script utilitário pra consultar os eventos gravados.

Uso:
    python view_events.py              # lista os últimos 50 eventos
    python view_events.py ABC1D23      # filtra por placa
"""
import sys
import sqlite3
import config


def main():
    conn = sqlite3.connect(config.DATABASE_PATH)
    c = conn.cursor()

    if len(sys.argv) > 1:
        plate = sys.argv[1].upper().replace('-', '').replace(' ', '')
        c.execute('''
            SELECT timestamp, plate_raw, plate_matched, confidence, event_type
            FROM events
            WHERE plate_matched = ?
            ORDER BY timestamp DESC
        ''', (plate,))
        print(f"\nEventos para a placa {plate}:\n")
    else:
        c.execute('''
            SELECT timestamp, plate_raw, plate_matched, confidence, event_type
            FROM events
            ORDER BY timestamp DESC
            LIMIT 50
        ''')
        print(f"\nÚltimos 50 eventos:\n")

    rows = c.fetchall()
    if not rows:
        print("Nenhum evento encontrado.")
        return

    print(f"{'Data/Hora':<20} {'OCR':<10} {'Placa':<10} {'Conf':>6}  Tipo")
    print("-" * 60)
    for ts, raw, matched, conf, tipo in rows:
        tipo_str = "ENTRADA" if tipo == 'entry' else "SAÍDA"
        print(f"{ts:<20} {raw:<10} {matched:<10} {conf:>5.1%}  {tipo_str}")

    conn.close()


if __name__ == '__main__':
    main()
