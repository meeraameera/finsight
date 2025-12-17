# FinSight

A **machine learning-powered personal finance tool** that automates transaction categorization and provides real-time insights into spending patterns.

---

## Problem Statement

Managing personal finances in the digital age is often tedious due to the high volume of transactions. Without automated tools, users struggle to manually categorize spending, leading to poor budget visibility and a lack of awareness of where their money is going.

---

## Project Goal

The goal of **Spending Analyzer** is to transform raw bank transaction data into **actionable financial insights**. By leveraging Machine Learning, the project eliminates manual bookkeeping and delivers a real-time, visual summary of a user’s financial life.

---

## Technical Approach

- **Data Engineering & Balancing:**  
  - Processed raw CSV transaction data.  
  - Implemented **Balanced Class Weighting** to ensure rare categories (e.g., Insurance, Income) are predicted accurately alongside frequent ones.

- **Pipeline Construction:**  
  - Developed a **Scikit-Learn Pipeline** combining custom text cleaners, TF-IDF feature extraction, and a Logistic Regression classifier into a single deployable object.

- **Model Tuning:**  
  - Performed **Grid Search** to optimize hyperparameters.  
  - Focused on **Macro-F1 Score** to guarantee high performance across all categories.

- **Backend API Development:**  
  - Built a **Flask server** with endpoints:  
    - `/categorize` for single-transaction predictions  
    - `/summary` for batch processing and analysis

- **Frontend State Management:**  
  - Implemented a **Streamlit dashboard** using `st.session_state` for file uploads and interactive filtering.  
  - Ensured smooth data exploration without losing session data.

---

## Impact

- **Model Performance:**  
  - **86% overall accuracy**  
  - **F1-score: 0.80** across all categories  
- Users can upload a standard CSV and immediately see a **prioritized list of spending habits and records**.  

---

## Key Challenges Faced

- **JSON Serialization:**  
  - Encountered `InvalidJSONError` due to `NaN` values.  
  - Resolved by coercing numeric types and filling nulls with `0.0` before API transmission.

- **Stateful UI:**  
  - Streamlit dashboard restarted on dropdown changes.  
  - Fixed by migrating data handling into **Session State**, maintaining user interactions.

---

## Installation & Execution

- Clone the repository.
- Create and activate a virtual environment
- Install required dependencies
- Execute the notebook `main.py` to train and fine-tune the model.
  - `python main.py`
- Connect to the API
  - `python backend.py`
- Run the Streamlit Web Application
  - `streamlit run streamlit.py`
