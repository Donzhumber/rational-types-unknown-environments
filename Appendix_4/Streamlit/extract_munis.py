import pandas as pd
import json

df = pd.read_csv('Data_CMH.csv')
# Get unique municipalities with their region and mean coordinates
muni_data = df.groupby('Municipio').agg({
    'Zona_Geografica': 'first',
    'Latitud': 'mean',
    'Longitud': 'mean'
}).reset_index()

# Filter out rows with NaN in important fields
muni_data = muni_data.dropna(subset=['Municipio', 'Zona_Geografica', 'Latitud', 'Longitud'])

# Convert to dictionary for the app
muni_dict = {}
for _, row in muni_data.iterrows():
    muni_dict[row['Municipio']] = {
        'region': row['Zona_Geografica'],
        'lat': row['Latitud'],
        'lon': row['Longitud']
    }

with open('muni_mapping.json', 'w') as f:
    json.dump(muni_dict, f, indent=4)

print(f"Mapped {len(muni_dict)} municipalities.")
