import streamlit as st
from streamlit_option_menu import option_menu

import Part1INT
import Part2INT
import Part3INT
import Part4INT

st.set_page_config(page_title="Projet Vision", layout="wide")

# Menu
with st.sidebar:
    selected = option_menu(
        menu_title="Menu Principal",
        options=["Partie 1", "Partie 2", "Partie 3", "Partie 4"],
        icons=["camera", "image", "graph-up", "gear"],
        menu_icon="menu-button",
        default_index=0
    )

if selected == "Partie 1":
    st.title("Partie 1 : Détection d'Objets")
    Part1INT.main()

elif selected == "Partie 2":
    st.title("Partie 2 : Calibrage de la caméra")
    Part2INT.main()

elif selected == "Partie 3":
    st.title("Partie 3 : Calcul de la position et de la distance d’un objet")
    Part3INT.main()

elif selected == "Partie 4":
    st.title("Partie 4 : Améliorations")
    Part4INT.main()