import pandas as pd
import json
import unicodedata

def normalize_name(value):
    if pd.isna(value): return ""
    text = unicodedata.normalize("NFKD", str(value).upper())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.replace(",", "").replace(".", "").strip()

# Strictly follow the regions defined in the app's fallback
DPTO_REGION_MAP = {
    "AMAZONAS": "Oriente/Selva",
    "ANTIOQUIA": "Andina",
    "ARAUCA": "Oriente/Selva",
    "ATLANTICO": "Caribe",
    "ATLÁNTICO": "Caribe",
    "BOGOTA DC": "Andina", 
    "BOGOTÁ, D.C.": "Andina",
    "BOLIVAR": "Caribe",
    "BOLÍVAR": "Caribe",
    "BOYACA": "Andina",
    "BOYACÁ": "Andina",
    "CALDAS": "Andina",
    "CAQUETA": "Oriente/Selva",
    "CAQUETÁ": "Oriente/Selva",
    "CASANARE": "Oriente/Selva",
    "CAUCA": "Pacífico/Roja",
    "CESAR": "Caribe",
    "CHOCO": "Pacífico/Roja",
    "CHOCÓ": "Pacífico/Roja",
    "CORDOBA": "Caribe",
    "CÓRDOBA": "Caribe",
    "CUNDINAMARCA": "Andina",
    "GUAINIA": "Oriente/Selva",
    "GUAINÍA": "Oriente/Selva",
    "GUAVIARE": "Oriente/Selva",
    "HUILA": "Andina",
    "LA GUAJIRA": "Caribe",
    "MAGDALENA": "Caribe",
    "META": "Oriente/Selva",
    "NARIÑO": "Pacífico/Roja",
    "NORTE DE SANTANDER": "Pacífico/Roja",
    "PUTUMAYO": "Oriente/Selva",
    "QUINDIO": "Andina",
    "QUINDÍO": "Andina",
    "RISARALDA": "Andina",
    "SAN ANDRES PROVIDENCIA Y SANTA CATALINA": "Caribe",
    "SAN ANDRÉS, PROVIDENCIA Y SANTA CATALINA": "Caribe",
    "ARCHIPIELAGO DE SAN ANDRES, PROVIDENCIA Y SANTA CATALINA": "Caribe",
    "SANTANDER": "Andina",
    "SUCRE": "Caribe",
    "TOLIMA": "Andina",
    "VALLE DEL CAUCA": "Pacífico/Roja",
    "VAUPES": "Oriente/Selva",
    "VAUPÉS": "Oriente/Selva",
    "VICHADA": "Oriente/Selva"
}

# Normalize the keys
NORM_DPTO_TO_REGION = {normalize_name(k): v for k, v in DPTO_REGION_MAP.items()}

# 12 Metropolitanos
METROS = [
    "BELLO", "BOGOTA DC", "BUCARAMANGA", "CARTAGENA DE INDIAS", "CUCUTA", 
    "ENVIGADO", "MANIZALES", "MEDELLIN", "PEREIRA", "SABANETA", 
    "SANTIAGO DE CALI", "SOACHA"
]

def get_region(row):
    muni = normalize_name(row.get('Municipio', ''))
    dpto = normalize_name(row.get('Departamento', ''))
    
    # Priority: Metros are always Metrópolis
    if muni in METROS:
        return "Metrópolis"
    
    # Otherwise: Strict Department mapping
    return NORM_DPTO_TO_REGION.get(dpto, "Sin región")

# Update Data_CMH.csv
df = pd.read_csv('Data_CMH.csv')
df['Zona_Geografica'] = df.apply(get_region, axis=1)
df.to_csv('Data_CMH.csv', index=False)

print("Data_CMH.csv fixed successfully.")
