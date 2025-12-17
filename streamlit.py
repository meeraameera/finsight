import pandas as pd
import streamlit as st
import requests
import plotly.express as px
import json


# --------------------------------------------------------------- 
# Configuration
# ---------------------------------------------------------------
flask_api_url = 'http://127.0.0.1:5000' 

st.set_page_config(layout="wide", page_title="Spending Analyzer")
st.title("Spending Analyzer")
st.markdown("Upload your transactions to categorize your spending and visualize category breakdown.")
st.divider()


# --------------------------------------------------------------- 
# State Initialization 
# ---------------------------------------------------------------
if 'analysis_complete' not in st.session_state:
    st.session_state['analysis_complete'] = False

if 'summary_df' not in st.session_state:
    st.session_state['summary_df'] = pd.DataFrame()

if 'full_transactions_df' not in st.session_state:
    st.session_state['full_transactions_df'] = pd.DataFrame()

if 'weekly_df' not in st.session_state:
    st.session_state['weekly_df'] = pd.DataFrame()


# --------------------------------------------------------------- 
# Helper Function for Robust DataFrame Creation
# ---------------------------------------------------------------
def safe_to_dataframe(data_list):
    """Converts list data to DataFrame, handling empty or invalid input safely."""

    if isinstance(data_list, list) and data_list:
        return pd.DataFrame(data_list)
    return pd.DataFrame() 


# --------------------------------------------------------------- 
# Main Data Processing Function 
# ---------------------------------------------------------------
def process_data_and_store_state(api_payload):
    """Calls the API and stores results in session state."""
    
    st.info("Sending transactions to the API for batch categorization and summarization...")
    
    try:
        response = requests.post(f"{flask_api_url}/summary", json=api_payload)
        
        if response.status_code == 200:
            response_data = response.json()
            
            # Extract and store spending summary
            st.session_state['summary_df'] = safe_to_dataframe(response_data.get('spending_summary', []))
            
            # Extract and store full transactions (for daily breakdown)
            st.session_state['full_transactions_df'] = safe_to_dataframe(response_data.get('categorized_transactions', []))
            if not st.session_state['full_transactions_df'].empty:
                st.session_state['full_transactions_df']['Date'] = st.session_state['full_transactions_df']['Date'].astype(str)
            
            # Extract and store weekly summary
            st.session_state['weekly_df'] = safe_to_dataframe(response_data.get('weekly_spending_ts', []))
            
            st.session_state['analysis_complete'] = True
            st.success("Analysis Complete! Data saved to session state.")
            st.rerun() 
            
        else:
            st.error(f"API ERROR: Failed to get summary. Status Code: {response.status_code}")
            st.code(json.dumps(response.json(), indent=2), language="json")
            
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Connection ERROR: Could not connect to the Flask API at **{flask_api_url}**. Please ensure 'app.py' is running.")


# --------------------------------------------------------------- 
# Upload Component and Input Validation 
# ---------------------------------------------------------------
st.header("1. Upload Transactions (CSV)")
uploaded_file = st.file_uploader(
    "Upload a CSV file with 'Date', 'Amount', and 'Description' columns", 
    type=['csv']
)

# Only run this block if a file is present and analysis hasn't completed or a new file is uploaded
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    
    # Replace NaN values in 'Amount' with 0.0 for JSON compliance 
    if 'Amount' in input_df.columns:
        input_df['Amount'] = pd.to_numeric(input_df['Amount'], errors='coerce').fillna(0.0) 
    
    # Input Validation
    required_columns = ['Date', 'Amount', 'Description']
    if not all(col in input_df.columns for col in required_columns):
        st.error(f"⚠️ Missing required columns in CSV. Must contain: **{required_columns}**")
        st.stop()
    
    st.dataframe(input_df.head(), use_container_width=True)
    
    # Prepare data for the API
    api_payload = []
    for index, row in input_df.iterrows():
        api_payload.append({
            "date": row['Date'], 
            "amount_sgd": row['Amount'], 
            "description": row['Description']
        })

    # Button triggers the processing function
    if st.button("Analyze Spending"):
        process_data_and_store_state(api_payload)


# --------------------------------------------------------------- 
# Display Results (Only run if state is complete)
# ---------------------------------------------------------------
if st.session_state['analysis_complete']:
    summary_df = st.session_state['summary_df']
    full_transactions_df = st.session_state['full_transactions_df']
    weekly_df = st.session_state['weekly_df']
    
    col1, col2 = st.columns(2)
    
    if not summary_df.empty:
        total_spent = summary_df['total_spent_sgd'].sum()
        num_categories = len(summary_df)
    else:
        total_spent = 0.0
        num_categories = 0
    
    with col1:
        st.metric(label="Total Analyzed Expenses", value=f"SGD {total_spent:,.2f}")
    with col2:
        st.metric(label="Number of Categories Found", value=num_categories)
        
    st.divider()
    
    # Display Category Breakdown Chart 
    st.header("2. Category Spending Breakdown")
    
    if not summary_df.empty and total_spent > 0:
        fig_pie = px.pie(
            summary_df[summary_df['total_spent_sgd'] > 0],
            values='total_spent_sgd', 
            names='category', 
            title='Spending Percentage by Category (SGD)',
            hole=.3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning("No spending data found to generate category chart.")
    
    # Detailed Category Summary Table 
    st.header("3. Detailed Category Summary Table")
    st.dataframe(summary_df, use_container_width=True)
    
    st.divider()
    
    # Display Time Series Data Breakdown (Vertical Stacked)
    st.header("4. Spending Breakdown by Time Period")
    
    # Daily Breakdown (Dropdown Filter)
    st.subheader("Daily Transaction Breakdown")
    
    if not full_transactions_df.empty:
        # Dropdown selection uses data stored in session_state
        unique_dates = sorted(full_transactions_df['Date'].unique(), reverse=True)
        
        selected_date = st.selectbox(
            "Select a Date to view transactions:",
            unique_dates,
            key='daily_date_select' 
        )
        
        # Filter transactions for the selected date
        daily_transactions = full_transactions_df[
            full_transactions_df['Date'] == selected_date
        ].sort_values(by='Amount', ascending=True)
        
        # Display table
        st.dataframe(
            daily_transactions[['Date', 'Amount', 'Description', 'Category']], 
            use_container_width=True
        )
    else:
        st.info("No transaction data available for daily breakdown.")
        
    st.markdown("---") 

    # Weekly Breakdown (Summary Table)
    st.subheader("Weekly Spending Summary")
    st.markdown("This shows **how much you spent in a week** by category, aggregated across the weeks found in your data.")
    
    if not weekly_df.empty:
        # Rename columns for better display
        display_df = weekly_df.rename(columns={
            'total_spent_sgd': 'Total Spent (SGD)',
            'date': 'Week Ending Date'
        })
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No weekly expenses found for table breakdown.")
        