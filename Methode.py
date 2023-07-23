import altair as alt
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import plotly.express as px
import networkx as nx
from prince import MCA
import plotly.graph_objects as go

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
            font=dict(color='white', size=20),
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
            font=dict(color='white', size=20),
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
            font=dict(color='white', size=10),
            bgcolor=None
        )

    # Configurer le layout du graphique
    fig.update_layout(
        width=1200,
        height=1600,
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    # Afficher le graphique
    
    return fig

# Example usage in a streamlit app



