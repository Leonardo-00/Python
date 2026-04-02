import chardet
import re

import json

def estrai_oggetti_json(testo):
    oggetti = []
    aperte = 0
    inizio = None

    for i, char in enumerate(testo):
        if char == '{':
            if aperte == 0:
                inizio = i
            aperte += 1
        elif char == '}':
            aperte -= 1
            if aperte == 0 and inizio is not None:
                oggetti.append(testo[inizio:i+1])
                inizio = None
    return oggetti

# 1. Leggi il file contenente gli oggetti JSON separati
with open("Stuff/sos.txt", "rb") as f:
    raw = f.read()
    encoding = chardet.detect(raw)['encoding']

with open("Stuff/sos.txt", "r", encoding=encoding, errors="ignore") as f:
    text = f.readline()

# 2. Estrai ogni oggetto JSON
blocchi_json = estrai_oggetti_json(text)

# 3. Convertili in oggetti Python (dizionari)
lista_dizionari = [json.loads(b) for b in blocchi_json]

# 4. Salva tutto in un file JSON valido
with open("output.json", "x", encoding="utf-8") as f_out:
    json.dump(lista_dizionari, f_out, indent=2, ensure_ascii=False)

print("✅ File JSON creato con successo!")