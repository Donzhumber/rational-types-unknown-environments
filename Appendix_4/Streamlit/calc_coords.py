import pandas as pd

df = pd.read_csv('Data_CMH.csv')
regions = df['Zona_Geografica'].unique()
print("Region, Lat, Lon")
for r in regions:
    subset = df[df['Zona_Geografica'] == r]
    mean_lat = subset['Latitud'].mean()
    mean_lon = subset['Longitud'].mean()
    print(f"{r}: [{mean_lat:.4f}, {mean_lon:.4f}]")
