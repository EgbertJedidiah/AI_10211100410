import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set page configuration as the first Streamlit command
st.set_page_config(
    page_title="ML & AI WORKBENCH",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import sections after setting the page config
from sections import home, regression, clustering, neural_network, llm_multimodal

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.selectbox("Choose a section:", (
    "Home", "Regression", "Clustering", "Neural Network", "LLM Q&A"
))

# Handling the selected section
if section == "Home":
    home.home_section()
elif section == "Regression":
    regression.regression_section()
elif section == "Clustering":
    clustering.clustering_section()
elif section == "Neural Network":
    neural_network.neural_network_section()
elif section == "LLM Q&A":
    llm_multimodal.llm_multimodal_section()

