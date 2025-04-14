import streamlit as st

import streamlit as st

def home_section():
    
    # Top Banner with gradient background
    st.markdown("""
    <div style="background: linear-gradient(to right, #4b6cb7, #182848); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; text-align: center;">🧠 ML & AI Workbench </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Introduction section
    st.markdown("""
    ### Welcome to the next generation of ML & AI exploration!
    
    This interactive platform provides a unified dashboard to explore, experiment with, and solve
    diverse Machine Learning and Artificial Intelligence problems. Perfect for students, researchers,
    and AI enthusiasts looking to deepen their understanding through hands-on practice.
    """)
    
    # Features section in columns
    st.markdown("## 📦 Core Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Data Analysis
        - **Regression Models**: Predict continuous values with customizable linear models
        - **Clustering Algorithms**: Discover patterns with K-Means clustering
        - **Interactive Visualizations**: Explore data in 2D and 3D spaces
        """)
        
    with col2:
        st.markdown("""
        ### 🤖 Advanced AI
        - **Neural Networks**: Build and train classification models with real-time metrics
        - **LLM Integration**: Leverage Gemini AI for document analysis and Q&A
        - **Computer Vision**: Process and analyze image data with modern techniques
        """)
    
    # How to use section
    st.markdown("## 🚀 Getting Started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Step 1
        Navigate through different tools using the sidebar menu
        """)
        #st.image("https://via.placeholder.com/150", caption="Navigation")
        
    with col2:
        st.markdown("""
        ### Step 2
        Upload your dataset or use our sample datasets to experiment
        """)
       # st.image("https://via.placeholder.com/150", caption="Data Upload")
        
    with col3:
        st.markdown("""
        ### Step 3
        Configure parameters, train models, and visualize results
        """)
       # st.image("https://via.placeholder.com/150", caption="Analysis")
    
    # Technology stack
    st.markdown("## 🛠️ Technology Stack")
    
    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
    
    with tech_col1:
        st.markdown("### Frontend")
        st.markdown("- Python Streamlit")
        st.markdown("- Interactive Widgets")
        st.markdown("- Responsive Design")
        
    with tech_col2:
        st.markdown("### ML Libraries")
        st.markdown("- Scikit-learn")
        st.markdown("- TensorFlow/Keras")
        st.markdown("- PyTorch")
        
    with tech_col3:
        st.markdown("### Visualization")
        st.markdown("- Plotly")
        st.markdown("- Matplotlib")
        st.markdown("- Seaborn")
        
    with tech_col4:
        st.markdown("### AI Integration")
        st.markdown("- Google Gemini API")
        st.markdown("- HuggingFace Models")
        st.markdown("- OpenCV")
    

    
    # Call to action
    st.markdown("## 🔍 Ready to Explore?")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Select a module from the sidebar to start your ML/AI journey. Whether you're a beginner or an experienced practitioner,
        our tools are designed to help you experiment, learn, and develop new insights.
        """)
    
   # with col2:
        #st.button("Start Exploring", type="primary")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <p>Designed and developed for Academic City University | Last Updated: April 2025</p>
        <p>Built with ❤️ using Python and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

    # Add session state initialization if needed
    if 'model_trained' not in st.session_state:
        st.session_state['model_trained'] = False