import streamlit as st
import pandas as pd
from Methode import plot_visiteurs,plot_transactions,plot_network,recommandation



def run_app():
    df = pd.read_csv('ouacompagny.csv')
    data_transactions = pd.read_csv('data_transactions.csv', parse_dates=['dates'])

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
    # Appeler la fonction plot_transactions() avec les données chargées
        fig = plot_transactions(data_transactions)

        st.plotly_chart(fig)
        st.markdown(" On divise en deux groupes : les transactions les jours de fête, et les transactions les autres jours. Nous avons ensuite utilisé le test de Student pour comparer les moyennes de ces deux groupes")
        if st.button("Interprétation"):    
            st.write("Il y a une différence significative entre les moyennes des transactions les jours de fête et les autres jours. Les jours de fête ont un impact significatif sur les transactions")
            st.write(f"T-statistique : 5.4")
            st.write(f"P-valeur : 0.0000")
            st.button("Cacher l'interprétation", key="effacer")
        
        else:
            st.write("Cliquez sur le bouton 'Interprétation' pour afficher les résultats.")

    def page2():
        regions = df['region_geographique'].unique()
        default_region = "Sénégal"
        selected_region = st.sidebar.selectbox('Choisir une région géographique', regions, index=regions.tolist().index(default_region))

        if selected_region != "Sénégal":
            st.error("Connectez-vous pour voir plus.")
        else:
            # Filtrer les données en fonction de la région géographique sélectionnée
            filtered_data = df[df['region_geographique'] == selected_region].reset_index(drop=True)

            # Afficher le graphique de réseau pour les données filtrées
            fig = plot_network(filtered_data)
            st.plotly_chart(fig)
    def page3():
        recommandation()
        
                
    menu = ["", "Accueil", "Historique Commandes", "Segmentation Clientèle","Recommandation"]
    choice = st.sidebar.selectbox("Navigation", menu)

    # Afficher la page en fonction du choix de l'utilisateur
    if choice == "":
        home()
    elif choice == "Accueil":
        home()
    elif choice == "Historique Commandes":
        page1()  
    elif choice == "Segmentation Clientèle":
        page2()      
    elif choice == "Recommandation":
        page3()  
run_app()
