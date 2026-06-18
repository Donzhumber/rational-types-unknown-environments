import pandas as pd
df = pd.read_csv('Data_CMH.csv')
caribe_dptos = ["ATLANTICO", "BOLIVAR", "CESAR", "CORDOBA", "LA GUAJIRA", "MAGDALENA", "SUCRE", "ARCHIPIELAGO"]

def is_caribe(dpto):
    d = str(dpto).upper()
    return any(c in d for c in caribe_dptos)

wrong = df[df.apply(lambda r: is_caribe(r['Departamento']) and r['Zona_Geografica'] == 'Andina', axis=1)]
print(f"Wrong rows: {len(wrong)}")
print(wrong[['Municipio', 'Departamento', 'Zona_Geografica']].head(20))
