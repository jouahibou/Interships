import pandas as pd
df = pd.read_csv('ouacompagny.csv')

regions = df['region_geographique'].unique()
#selected_region = st.sidebar.selectbox('Choisir une région géographique', regions)

# Filtrer les données en fonction de la région géographique sélectionnée
filtered_data = df[df['region_geographique'] == "Sénégal"]

print(len(filtered_data))

for i in range(len(df)):
    print(df.loc[i, 'nom_produit'], df.loc[i, 'type_client'])
    print(df.loc[i, 'nom_produit'], df.loc[i, 'region_geographique'])
