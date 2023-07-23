import altair as alt
import pandas as pd
import numpy as np
from scipy import stats
#import locale
import plotly.express as px
import networkx as nx
from prince import MCA
import plotly.graph_objects as go
import streamlit as st


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

def plot_transactions():
    #locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
    start_date = pd.to_datetime('2022-04-01')
    end_date = pd.to_datetime('2023-05-1')
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
    dates = pd.date_range(start_date, end_date)

    # Générer des données de transactions aléatoires pour chaque jour
    transactions = np.random.randint(40, 600, size=len(dates))

    # Ajouter des pics pour chaque jour de fête
    for holiday in holidays:
        start_index = (holiday - start_date).days - 7
        end_index = (holiday - start_date).days + 7
        transactions[start_index:end_index] += np.random.randint(5, 150)

    # Créer un DataFrame avec les données
    data = pd.DataFrame({'dates': dates, 'transactions': transactions})

    # Créer le graphique avec Plotly
    fig = px.line(data, x='dates', y='transactions', title='Transactions')
    fig.update_xaxes(title='Date')
    fig.update_yaxes(title=' Nombre de Transactions')

    # Ajouter des marqueurs pour les jours de fête
    for holiday in holidays:
        fig.add_vline(x=holiday, line_color="red", line_dash="dash")
        fig.add_annotation(x=holiday, y=transactions.max(), text=holiday.strftime('%d %b %Y'), showarrow=False, yshift=10)

    return fig

def analyze_transactions():
    start_date = pd.to_datetime('2022-04-01')
    end_date = pd.to_datetime('2023-05-1')
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
    dates = pd.date_range(start_date, end_date)

    # Générer des données de transactions aléatoires pour chaque jour
    transactions = np.random.randint(50, 600, size=len(dates))

    # Ajouter des pics pour chaque jour de fête
    for holiday in holidays:
        if start_date <= holiday <= end_date:
            start_index = (holiday - start_date).days - 7
            end_index = (holiday - start_date).days + 7
            transactions[start_index:end_index] += np.random.randint(500, 2000)

    # Créer un DataFrame avec les données
    data = pd.DataFrame({'dates': dates, 'transactions': transactions})

    # Créer une nouvelle colonne indiquant si c'est un jour de fête
    data['is_holiday'] = data['dates'].apply(lambda x: x in holidays)

    # Diviser les données en deux groupes
    holiday_transactions = data.loc[data['is_holiday'], 'transactions']
    non_holiday_transactions = data.loc[~data['is_holiday'], 'transactions']

    # Effectuer le test de Student
    t_stat, p_value = stats.ttest_ind(holiday_transactions, non_holiday_transactions)

    # Retourner les résultats du test
    return t_stat, p_value






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

    # Créer un DataFrame des arêtes
    edges = pd.DataFrame(list(G.edges()), columns=['source', 'target'])

    # Créer un graphique de réseau interactif avec Plotly
    fig = go.Figure()

    # Add nodes to the plot
    fig.add_trace(go.Scatter(
        x=nodes['x'], 
        y=nodes['y'],
        mode='markers',
        marker=dict(
            size=10,
            color='blue',
            symbol='circle'
        ),
        text=nodes['node'],
        hoverinfo='text'
    ))

    # Add edges to the plot
    for i, edge in edges.iterrows():
        fig.add_trace(go.Scatter(
            x=[nodes.loc[nodes['node'] == edge['source'], 'x'].iloc[0], 
               nodes.loc[nodes['node'] == edge['target'], 'x'].iloc[0]],
            y=[nodes.loc[nodes['node'] == edge['source'], 'y'].iloc[0], 
               nodes.loc[nodes['node'] == edge['target'], 'y'].iloc[0]],
            mode='lines',
            line=dict(
                color='gray',
                width=1
            ),
            hoverinfo='skip',
        ))

    fig.update_layout(
        width=600,
        height=600,
        showlegend=False,
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0
        ),
        hovermode='closest'
    )

    return fig


# Example usage in a streamlit app
