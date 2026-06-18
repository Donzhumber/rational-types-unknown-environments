import pandas as pd
df = pd.read_csv('Data_CMH.csv')
dupes = df.groupby('CódigoDANEdeMunicipio')['Zona_Geografica'].nunique()
print(dupes[dupes > 1])
