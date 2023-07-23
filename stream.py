import streamlit as st
import pandas as pd
from Methode import plot_visiteurs,plot_transactions,analyze_transactions,plot_network

df = pd.read_csv('ouacompagny.csv')

def home():
    st.title('Oua_compagny')
    st.markdown("Le but de ce projet est de développer un système de recommandation pour un site de commerce électronique en utilisant les données clients et produits existantes. Le système de recommandation permettra de recommander des produits similaires ou complémentaires à ceux que les clients ont achetés ou consultés dans le passé. Cela peut aider à augmenter les ventes en suggérant des produits pertinents aux clients et à améliorer l'expérience utilisateur en offrant des recommandations personnalisées.")
    st.write("  ")    
    st.write('graphique des visiteurs :')
    chart = plot_visiteurs()
    st.altair_chart(chart)
    if st.button("Interprétation"):
        st.write("On remarque que 86 % des visiteurs sont de nouveaux visiteurs contre 14 % . Dans l'optique de comprendre comment les visiteurs interagissent avec le site on va essayer de voir les catégories de produits les plus populaires auprés des nouveaux clients ainsi que les taux de conversions des différents pages ")
        st.button("Cacher l'interprétation", key="effacer")        

def page1():
    st.title('Visualisation des transactions')
    fig = plot_transactions()
    st.plotly_chart(fig)
    st.markdown(" On divise en deux groupes : les transactions les jours de fête, et les transactions les autres jours. Nous avons ensuite utilisé letest de Student pour comparer les moyennes de ces deux groupes")
    t_stat, p_value = analyze_transactions()
    if st.button("Interprétation"):
        if p_value < 0.05:
            st.write("Il y a une différence significative entre les moyennes des transactions les jours de fête et les autres jours. Les jours de fête ont un impact significatif sur les transactions")
            st.write(f"T-statistique : 5.4")
            st.write(f"P-valeur : {p_value:.4f}")
            st.button("Cacher l'interprétation", key="effacer")
        else:
            st.write("Il n'y a pas de différence significative entre les moyennes des transactions les jours de fête et les autres jours.")
    else:
        st.write("Cliquez sur le bouton 'Interprétation' pour afficher les résultats.")
    
def page2():
# Afficher le graphique de réseau interactif
    network_fig = plot_network(df)
    st.plotly_chart(network_fig)
    


menu = ["Accueil", "Page 1","Page 2"]
choice = st.sidebar.selectbox("Navigation", menu)

# Afficher la page en fonction du choix de l'utilisateur
if choice == "Accueil":
    home()
elif choice == "Page 1":
    page1()  
elif choice == "Page 2":
    page2()      