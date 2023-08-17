import altair as alt
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import plotly.express as px
import networkx as nx
from prince import MCA
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from prince import MCA 
def plot_visiteurs():
    # Donnees
    labels = ['Nouveaux visiteurs', 'Clients existants']
    sizes = [86, 14]
    colors = ['#ff9999','#66b3ff']

    # Cration d'un DataFrame avec les données
    data = pd.DataFrame({'labels': labels, 'sizes': sizes})

    # Création du graphique avec Altair
    chart = alt.Chart(data).mark_bar().encode(
        #x=alt.X('stars', axis=alt.Axis(labelAngle=-45))
        x=alt.X('labels:N', title=None,axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('sizes:Q', title='%'),
        color=alt.Color('labels:N', scale=alt.Scale(range=colors), legend=None)
    ).properties(
        width=500,
        height=300,
        title='Répartition des visiteurs entre nouveaux et anciens clients'
    )

    return chart

def plot_transactions(data_transaction):
    holidays = [
        pd.to_datetime('2022-04-24'),  # Une semaine avant Pâques
        pd.to_datetime('2022-04-30'),  # Une semaine avant 1er mai
        pd.to_datetime('2022-06-05'),  # Une semaine avant 6 juin
        pd.to_datetime('2022-07-04'),
        pd.to_datetime('2022-08-04'),
        pd.to_datetime('2022-09-04'),# Usne semaine avant 11 juillet
        pd.to_datetime('2022-12-18'),  # Une semaine avant Noël
        pd.to_datetime('2022-12-24'),  # Une semaine avant 1er janvier
        pd.to_datetime('2023-04-02'),  # Une semaine avant Pâques
        pd.to_datetime('2023-04-15')   # Une semaine avant 22 avril
    ]

    fig = px.line(data_transaction, x='dates', y='transactions', title='Transactions')
    fig.update_xaxes(title='Date')
    fig.update_yaxes(title='Nombre de Transactions')

    # Ajouter des marqueurs pour les jours de fête
    for holiday in holidays:
        fig.add_vline(x=holiday, line_color="red", line_dash="dash")
        fig.add_annotation(x=holiday, y=data_transaction.max().loc['transactions'], text=holiday.strftime('%d %b %Y'), showarrow=False, yshift=10)

    return fig

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from prince import MCA


import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from prince import MCA

def plot_network(df):
    # Sélection des variables catégorielles pour l'ACM
    cat_vars = ["nom_produit", "type_action", "type_client", "region_geographique"]

    # Convertir les variables catégorielles en catégories
    for var in cat_vars:
        df[var] = df[var].astype('category')

    # Instancier la classe MCA et ajuster le modèle
    mca = MCA(n_components=2)
    mca.fit(df[cat_vars])

    # Créer un graphique de réseau avec NetworkX
    G = nx.Graph()
    for i in range(len(df)):
        G.add_edge(df.loc[i, 'nom_produit'], df.loc[i, 'type_client'])
        G.add_edge(df.loc[i, 'nom_produit'], df.loc[i, 'region_geographique'])

    # Calculer les positions des nœuds avec NetworkX
    pos = nx.spring_layout(G)

    # Créer un DataFrame des positions des nœuds
    nodes = pd.DataFrame(pos, index=['x', 'y']).transpose().reset_index()
    nodes.rename(columns={'index': 'node'}, inplace=True)

    # Ajouter les informations de texte pour chaque nœud
    nodes['node_text'] = nodes['node']

    # Ajouter une colonne "size" pour les nœuds type_client et region_geographique
    nodes.loc[nodes['node'].isin(df['type_client'].unique()), 'size'] = 10
    nodes.loc[nodes['node'].isin(df['region_geographique'].unique()), 'size'] = 10

    # Ajouter une colonne "size" pour les nœuds nom_produit
    nodes['size'] = 100

    # Créer un DataFrame des arêtes
    edges = pd.DataFrame(list(G.edges()), columns=['source', 'target'])
    edges['text'] = edges['source'] + ' - ' + edges['target']

    # Créer un graphique de réseau interactif avec Plotly
    fig = go.Figure()

    node_trace_type_client = go.Scatter(
        x=nodes[nodes['node'].isin(df['type_client'].unique())]['x'],
        y=nodes[nodes['node'].isin(df['type_client'].unique())]['y'],
        mode='markers',
        marker=dict(
            size=25,
            color='blue',
            opacity=0.8,
            line=dict(width=0.5, color='black')
        ),
        text=nodes[nodes['node'].isin(df['type_client'].unique())]['node_text'],
        hoverinfo='text+name',
        textfont=dict(size=10),
        textposition='middle center',
        name='type_client'
    )

    node_trace_region = go.Scatter(
        x=nodes[nodes['node'].isin(df['region_geographique'].unique())]['x'],
        y=nodes[nodes['node'].isin(df['region_geographique'].unique())]['y'],
        mode='markers',
        marker=dict(
            size=50,
            color='green',
            opacity=0.8,
            line=dict(width=0.5, color='black')
        ),
        text=nodes[nodes['node'].isin(df['region_geographique'].unique())]['node_text'],
        hoverinfo='text+name',
        textfont=dict(size=10),
        textposition='middle center',
        name='region_geographique'
    )

    node_trace_produit = go.Scatter(
        x=nodes[nodes['node'].isin(df['nom_produit'].unique())]['x'],
        y=nodes[nodes['node'].isin(df['nom_produit'].unique())]['y'],
        mode='markers',
        marker=dict(
            size=8,
            color='red',
            opacity=0.8,
            line=dict(width=0.5, color='black'),
            
        ),
        text=nodes[nodes['node'].isin(df['nom_produit'].unique())]['node_text'],
        hoverinfo='text+name',
        textfont=dict(size=10),
        textposition='middle center',
        name='Nom du produit'
    )

    # Ajouter les arêtes au graphique
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    fig.add_trace(node_trace_type_client)
    fig.add_trace(node_trace_region)
    fig.add_trace(node_trace_produit)
    fig.add_trace(edge_trace)

    # Ajouter les annotations pour les nœuds type_client
    for node in nodes[nodes['node'].isin(df['type_client'].unique())]['node']:
        x, y = nodes[nodes['node'] == node][['x', 'y']].values[0]
        fig.add_annotation(
            x=x,
            y=y,
            xref="x",
            yref="y",
            text=node,
            showarrow=False,
            font=dict(color='brown', size=20),
            bgcolor=None
        )

    # Ajouter les annotations pour les nœuds region_geographique
    for node in nodes[nodes['node'].isin(df['region_geographique'].unique())]['node']:
        x, y = nodes[nodes['node'] == node][['x', 'y']].values[0]
        fig.add_annotation(
            x=x,
            y=y,
            xref="x",
            yref="y",
            text=node,
            showarrow=False,
            font=dict(color='brown', size=20),
            bgcolor=None
        )

    # Ajouter les annotations pour les nœuds nom_produit
    for node in nodes[nodes['node'].isin(df['nom_produit'].unique())]['node']:
        x, y = nodes[nodes['node'] == node][['x', 'y']].values[0]
        fig.add_annotation(
            x=x,
            y=y,
            xref="x",
            yref="y",
            text=node,
            showarrow=False,
            font=dict(color='blue', size=10),
            bgcolor=None
        )

    # Configurer le layout du graphique
    fig.update_layout(
        width=1000,
        height=1400,
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[],
        font=dict(color='blue', size=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    fig.update_layout(
    title_text='Réseau de produits, clients et régions géographiques',
    title_font=dict(color='blue', size=30)
)

    # Afficher le graphique
    
    return fig

import streamlit as st
import sys
import pandas as pd

# Chargement des données

def recommandation():
    data = pd.read_csv("ouacompagny.csv")

# Traitement des données utilisateur
    user_data = data[['type_client', 'region_geographique']]

# Appliquer l'Analyse en Correspondances Multiples (MCA) pour obtenir les coordonnées des utilisateurs dans l'espace des composantes principales
    mca = MCA(n_components=2)  # Utilisation de MCA au lieu de PCA
    
# Traitement des données articles
    article_data = data[['category', 'nom_produit']]
# Appliquer l'Analyse en Correspondances Multiples (MCA) pour obtenir les coordonnées des articles dans l'espace des composantes principales
    article_coords = mca.fit_transform(article_data)

# Calcul des similarités entre articles (exemple : similarité cosinus)
    article_similarities = cosine_similarity(article_coords)
    # Entrez le type de client
    client_type = st.radio("Type de client", ['Nouveaux visiteurs', 'Clients existants'])
    # Entrez le pays
    country = st.selectbox("Pays", ['Sénégal'])

    # Entrez le nom du produit
    product_name = st.text_input("Nom du produit")

    recommendations = None

# Utilisez les valeurs récupérées pour exécuter le code
    if st.button("Recommendations"):
    # Demander le type de client à l'utilisateur
        user_type_client = client_type

    # Demander la région géographique à l'utilisateur
        user_region_geographique = country

        matching_products = data[data['nom_produit'].str.contains(product_name, case=False)]
        if not matching_products.empty:
            product_index = matching_products.index[0]
            product_value = matching_products['nom_produit'].iloc[0]
            st.success(product_value)
            # Trouver les articles similaires seulement si le produit est trouvé
            similar_articles = article_similarities[product_index]

            # Créer une série pandas avec les indices des articles et leurs similarités
            similar_articles_series = pd.Series(similar_articles, index=data.index)

            # Trier les articles par similarité en ordre décroissant et sélectionner les 4 premiers
            top_4_articles_indices = similar_articles_series.sort_values(ascending=False).head(4).index

            # Récupérer les informations des produits similaires correspondant aux indices sélectionnés
            similar_products = data.loc[top_4_articles_indices, ['nom_produit', 'type_client', 'region_geographique']]

            # Filtrer les produits similaires pour exclure le produit donné
            similar_products = similar_products[similar_products['nom_produit'] != product_name]

            # Supprimer les doublons des produits similaires
            similar_products = similar_products.drop_duplicates(subset='nom_produit')

            # Stocker les recommandations dans la variable déclarée précédemment
            recommandations = similar_products['nom_produit']

        else:
             st.error("Ce produit n'existe pas")

# Afficher les recommandations si elles existent
    if recommandations is not None:
        st.write(recommandation)


