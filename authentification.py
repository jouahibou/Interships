import streamlit as st
import webbrowser


hide_menu_style = """
    <style>
    #MainMenu {display: display;}
    </style>
"""

# Afficher la feuille de style CSS
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Create an empty container
import streamlit as st

# Insert a form in the container for "Invité" and "Connexion" buttons
with st.form("login"):
    st.write("Bienvenue, vous êtes connecté en tant qu'invité")
    st.write("Veuillez vous connecter :")
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")
    submit_button = st.form_submit_button("Connexion")

# Vérifier si l'utilisateur a cliqué sur le bouton "Connexion"
if submit_button:
    # Vérifier si l'utilisateur a entré les bonnes informations d'identification
    if username == 'adminjjjjjjouahibou' and password == 'passwordjjjjjjjouahibou':
        st.success("Connexion réussie!")
    else:
        st.warning("Identifiant ou mot de passe incorrect.")



