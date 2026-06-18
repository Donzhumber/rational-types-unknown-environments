import pandas as pd
import json
import unicodedata

def normalize_name(value):
    if pd.isna(value): return ""
    text = unicodedata.normalize("NFKD", str(value).upper())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.replace(",", "").replace(".", "").strip()

# Centralized Strict Mapping
DPTO_REGION_MAP = {
    # Andina
    "ANTIOQUIA": "Andina", "BOYACA": "Andina", "CALDAS": "Andina", 
    "CUNDINAMARCA": "Andina", "HUILA": "Andina", "QUINDIO": "Andina", 
    "RISARALDA": "Andina", "SANTANDER": "Andina", "TOLIMA": "Andina",
    "BOGOTA DC": "Andina", "BOGOTA D C": "Andina",
    
    # Caribe
    "ATLANTICO": "Caribe", "BOLIVAR": "Caribe", "CESAR": "Caribe", 
    "CORDOBA": "Caribe", "LA GUAJIRA": "Caribe", "MAGDALENA": "Caribe", 
    "SUCRE": "Caribe", "SAN ANDRES PROVIDENCIA Y SANTA CATALINA": "Caribe",
    "ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y SANTA CATALINA": "Caribe",
    
    # Pacífica / Zona Roja
    "CAUCA": "Pacífico/Roja", "CHOCO": "Pacífico/Roja", 
    "NARIÑO": "Pacífico/Roja", "NORTE DE SANTANDER": "Pacífico/Roja", 
    "VALLE DEL CAUCA": "Pacífico/Roja",
    
    # Oriente / Selva
    "AMAZONAS": "Oriente/Selva", "ARAUCA": "Oriente/Selva", 
    "CAQUETA": "Oriente/Selva", "CASANARE": "Oriente/Selva", 
    "GUAINIA": "Oriente/Selva", "GUAVIARE": "Oriente/Selva", 
    "META": "Oriente/Selva", "PUTUMAYO": "Oriente/Selva", 
    "VAUPES": "Oriente/Selva", "VICHADA": "Oriente/Selva"
}

METROS = [
    "BELLO", "BOGOTA DC", "BUCARAMANGA", "CARTAGENA DE INDIAS", "CUCUTA", 
    "ENVIGADO", "MANIZALES", "MEDELLIN", "PEREIRA", "SABANETA", 
    "SANTIAGO DE CALI", "SOACHA"
]

def get_region(row):
    muni = normalize_name(row.get('Municipio', ''))
    dpto = normalize_name(row.get('Departamento', ''))
    if muni in METROS:
        return "Metrópolis"
    norm_dpto = normalize_name(dpto)
    return DPTO_REGION_MAP.get(norm_dpto, "Sin región")

# Update Data_CMH.csv
df = pd.read_csv('Data_CMH.csv')
df['Zona_Geografica'] = df.apply(get_region, axis=1)
df.to_csv('Data_CMH.csv', index=False)
print("Data_CMH.csv updated.")
