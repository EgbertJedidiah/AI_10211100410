## 🧾 Credits

Created by **EGBERT JEDIDIAH OWIREDU OSEI** with roll number **10211100410** 
---

# 🧠 ML & AI WORKBENCH

A dynamic and user-friendly Streamlit application built to explore core Machine Learning and AI techniques using real-world data.
From classic regression models to multimodal LLM-powered Q&A, this tool helps users interactively dive into AI.

---
🚀 What You Can Do
📈 Regression Module – Forecast numeric outcomes using Linear Regression.

🔍 Clustering Tool – Use K-Means to uncover natural groupings in data with rich visualizations.

🧠 Neural Networks – Train simple feedforward models for classification.

🤖 LLM-Based Q&A – Ask questions across different formats (PDFs, CSVs, plain text) using Google's Gemini AI.
---

## 📁 Directory Structure

```
ML_AI_Explorer/
├── app.py
├── .env
├── README.md
├── requirements.txt
├── datasets/
│   ├── Ghana_Election_Result.csv
│   ├── 2025-Budget-Statement-and-Economic-Policy_v4.pdf
│   └── handbook.pdf
├── sections/
│   ├── __init__.py
│   ├── home.py
│   ├── regression.py
│   ├── clustering.py
│   ├── neural_network.py
│   └── llm_multimodal.py
├── images/
│   └── llm_architecture.png
---

🖼 LLM Q&A Flow
User selects a document (CSV or PDF).

#The content is parsed and embedded as context.

#A user question is appended to this context.

#The Gemini API processes the input and responds.

#The answer is shown in real-time within the app.






## 📊 Datasets Used in LLM Q&A

| Dataset Name          | Type |
|-----------------------|------|
| Ghana Election Results| CSV  | 
| 2025 Budget Statement | PDF  |  
| Academic City Handbook| PDF  |


⚙️ Setup Guide
# 1. Clone the repository
git clone https://github.com/yourusername/ML_AI_Workbench.git
cd ML_AI_Workbench

# 2. Set up virtual environment
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

▶️ Launch the App
streamlit run app.py

 🧠 LLM Q&A Flow

Below is a diagram showing how the Q&A system works using Google Gemini AI:
![LLM Q&A Flow](images/llm_architecture.png)




📜 License
This project is open-sourced under the MIT License.