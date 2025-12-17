import os
import re 
import pandas as pd
import pickle

from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask, request, jsonify


app = Flask(__name__)
model_filename = 'transactions_categorizer_model.pkl'


# --------------------------------------------------------------- 
# Custom Text Cleaner Transformer 
# --------------------------------------------------------------- 
class AdvancedTextCleaner(BaseEstimator, TransformerMixin):
    """A custom transformer to clean and simplify text data."""

    def fit(self, X, y=None):
        """
        Fits the transformer.

        Args
            X (list or array-like of str): The input data.
            y (None): Included for pipeline compatibility.

        Returns
            self: Returns the instance itself.
        """
        return self


    def transform(self, X):
        """
        Cleans and simplifies the input text data.

        Args:
            X (list or array-like of str): The input text data to be cleaned.

        Returns:
            list: The list of cleaned and simplified text strings.
        """
        cleaned_text = []

        for text in X:
            text = str(text)
            text = text.upper().strip()                     # Convert to uppercase and strip whitespace
            text = re.sub(r'[#\*]+[0-9A-Z]+', ' ', text)    # Remove transaction-specific noise (e.g., order numbers, hash tags)
            text = re.sub(r'[^A-Z\s-]', ' ', text)          # Remove digits and punctuation (except internal hyphens, which might be useful)
            text = re.sub(r'\s+', ' ', text).strip()        # Collapse multiple spaces into a single space
            
            cleaned_text.append(text)
        return cleaned_text


# --------------------------------------------------------------- 
# Load the Model
# --------------------------------------------------------------- 
model = None
try:
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model loaded successfully from {model_filename}")
except FileNotFoundError:
    print(f"ERROR: {model_filename} not found. Ensure the training phase is complete.")


# Mock data load remains as a helpful check/template
try:
    mock_df = pd.read_csv('transaction_records.csv')
    mock_df = mock_df[['Date', 'Amount']].copy()
    print("✅ Mock Data structure loaded for aggregation.")
except FileNotFoundError:
    print("WARNING: transaction_records.csv not found. Summary endpoint may fail.")


@app.route('/', methods=['GET'])
def home():
    return "Transaction Categorizer API is running! Access /categorize or /summary."


# ---------------------------------------------------------------
# Helper Function for Category Aggregation 
# ---------------------------------------------------------------
def get_spending_summary(df):
    """Groups the categorized transactions and calculates the total spending."""

    expenses = df[df['Amount'] < 0].copy()
    expenses['Absolute_Amount'] = expenses['Amount'].abs()
    spending_summary = expenses.groupby('Category')['Absolute_Amount'].sum().sort_values(ascending=False)
    
    total_spending = spending_summary.sum()
    spending_percentages = (spending_summary / total_spending * 100).round(2)
    
    summary_list = []
    for category, total in spending_summary.items():
        summary_list.append({
            "category": category,
            "total_spent_sgd": round(total, 2),
            "percentage": round(spending_percentages[category], 2)
        })
    return summary_list


# ---------------------------------------------------------------
# Helper Function for Time-Series Aggregation 
# ---------------------------------------------------------------
def get_time_series_summary(df, freq='D'):
    """Groups categorized expenses by date and category, then resamples."""
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce') 
    df = df.dropna(subset=['Date']) # Drop rows where date conversion failed
    df = df.set_index('Date')

    # Filter for expenses and take absolute value
    expenses = df[df['Amount'] < 0].copy()
    expenses['Amount'] = expenses['Amount'].abs()

    # Resample/Group
    spending_ts = expenses.groupby('Category')['Amount'].resample(freq).sum().fillna(0)

    # Format for JSON output
    summary_list = []
    for (category, date), total in spending_ts.items():
        if total > 0:
            summary_list.append({
                "date": date.strftime('%Y-%m-%d'),
                "category": category,
                "total_spent_sgd": round(total, 2),
                "frequency": freq
            })
    return summary_list


# ---------------------------------------------------------------
# The low-latency endpoint for a single transaction
# ---------------------------------------------------------------
@app.route('/categorize', methods=['POST'])
def categorize():
    if model is None:
        return jsonify({"ERROR": "Model not loaded. Server configuration error."}), 500

    data = request.get_json()
    description = data.get('description', '')
    
    if not description:
        return jsonify({"ERROR": "Missing 'description' parameter"}), 400

    try:
        prediction = model.predict([description])[0]
        
        return jsonify({
            "description": description.upper().strip(), 
            "predicted_category": prediction
        })
    except Exception as e:
        return jsonify({"ERROR": f"Prediction failed: {e}"}), 500
    

# ---------------------------------------------------------------    
# The high-value endpoint for batch processing and aggregation 
# ---------------------------------------------------------------  
@app.route('/summary', methods=['POST'])
def summarize_transactions():
    if model is None:
        return jsonify({"ERROR": "Model not loaded. Server configuration error."}), 500
    
    transactions = request.get_json()
    if not transactions or not isinstance(transactions, list):
        return jsonify({"ERROR": "Input must be a list of transactions."}), 400
    # Prepare data for batch processing
    descriptions = [t.get('description', '') for t in transactions]
    amounts = [t.get('amount_sgd') for t in transactions]
    dates = [t.get('date', '') for t in transactions] 
    
    # Batch Categorization (Prediction)
    try:
        predicted_categories = model.predict(descriptions)
    except Exception as e:
        return jsonify({"ERROR": f"Batch prediction failed: {e}"}), 500

    # Reconstruct DataFrame for Aggregation 
    categorized_df = pd.DataFrame({
        'Date': dates,
        'Amount': amounts,
        'Description': descriptions, 
        'Category': predicted_categories
    })
    
    # Calculate Summaries and prepare raw transaction list
    category_summary = get_spending_summary(categorized_df.copy())
    
    weekly_spending = get_time_series_summary(categorized_df.copy(), freq='W')
    
    full_transactions = categorized_df.to_dict('records')

    return jsonify({
        "status": "success",
        "total_transactions": len(transactions),
        "spending_summary": category_summary,
        "weekly_spending_ts": weekly_spending, 
        "categorized_transactions": full_transactions 
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    