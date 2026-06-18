import json
import unicodedata

def normalize_name(value):
    if not value: return ""
    text = unicodedata.normalize("NFKD", str(value).upper())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.replace(",", "").replace(".", "").strip()

DPTO_REGION_FALLBACK_RAW = {
    # Andina
    "ANTIOQUIA": "Andina", "BOYACÁ": "Andina", "CALDAS": "Andina", 
    "CUNDINAMARCA": "Andina", "HUILA": "Andina", "QUINDÍO": "Andina", 
    "RISARALDA": "Andina", "SANTANDER": "Andina", "TOLIMA": "Andina",
    "BOGOTÁ, D.C.": "Andina",
    
    # Caribe
    "ATLÁNTICO": "Caribe", "BOLÍVAR": "Caribe", "CESAR": "Caribe", 
    "CÓRDOBA": "Caribe", "LA GUAJIRA": "Caribe", "MAGDALENA": "Caribe", 
    "SUCRE": "Caribe", "SAN ANDRÉS, PROVIDENCIA Y SANTA CATALINA": "Caribe",
    "ARCHIPIELAGO DE SAN ANDRES, PROVIDENCIA Y SANTA CATALINA": "Caribe",
    
    # Pacífica / Zona Roja
    "CAUCA": "Pacífica / Zona Roja", "CHOCÓ": "Pacífica / Zona Roja", 
    "NARIÑO": "Pacífica / Zona Roja", "NORTE DE SANTANDER": "Pacífica / Zona Roja", 
    "VALLE DEL CAUCA": "Pacífica / Zona Roja",
    
    # Oriente / Selva
    "AMAZONAS": "Oriente / Selva", "ARAUCA": "Oriente / Selva", 
    "CAQUETÁ": "Oriente / Selva", "CASANARE": "Oriente / Selva", 
    "GUAINÍA": "Oriente / Selva", "GUAVIARE": "Oriente / Selva", 
    "META": "Oriente / Selva", "PUTUMAYO": "Oriente / Selva", 
    "VAUPÉS": "Oriente / Selva", "VICHADA": "Oriente / Selva"
}

DPTO_REGION_FALLBACK = {
    normalize_name(departamento): region
    for departamento, region in DPTO_REGION_FALLBACK_RAW.items()
}

with open('co_2018_MGN_MPIO_POLITICO.geojson', 'r') as f:
    geojson = json.load(f)

inconsistent = []
for feature in geojson['features']:
    props = feature['properties']
    dpto = props.get('DPTO_CNMBR', '')
    muni = props.get('MPIO_CNMBR', '')
    norm_dpto = normalize_name(dpto)
    region = DPTO_REGION_FALLBACK.get(norm_dpto, "Sin región")
    
    if region == "Sin región":
        inconsistent.append((dpto, muni))

print(f"Total features: {len(geojson['features'])}")
print(f"Inconsistent departments: {len(inconsistent)}")
for d, m in inconsistent[:20]:
    print(f"Dpto: {d}, Muni: {m}")
