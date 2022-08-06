import streamlit as st
from multiapp import MultiApp
from apps import home, model_random_forest, model_svm, model_regresion_logistica, model_lib_prophet # import your app modules here

app = MultiApp()

st.markdown("""
#  Inteligencia de Negocios - Equipo A 
""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Modelo Random Forest", model_random_forest.app)
app.add_app("Modelo Regresión Logística", model_regresion_logistica.app)
app.add_app("Modelo SVM de Regresión", model_svm.app)
app.add_app("Modelo basado en la librería Prophet", model_lib_prophet.app)
# The main app
app.run()