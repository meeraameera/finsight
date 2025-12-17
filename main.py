import pandas as pd
import re
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


# --------------------------------------------------------------- 
# Custom Text Cleaner Transformer (Ensures consistency)
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
# Load Data & Initial Prep
# --------------------------------------------------------------- 
df = pd.read_csv('transaction_records_train.csv')

X = df['Description']
y = df['Category']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --------------------------------------------------------------- 
# Create the Base Pipeline (With Balanced Weighting)
# --------------------------------------------------------------- 
text_classifier_pipeline = Pipeline([
    ('clean', AdvancedTextCleaner()), 
    ('tfidf', TfidfVectorizer(stop_words='english')), 
    ('clf', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'))
])


# --------------------------------------------------------------- 
# Hyperparameter Tuning using Grid Search 
# --------------------------------------------------------------- 
param_grid = {
    # Included up to (1, 3) N-grams for better context capture
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)], 
    
    # Wider range of C for fine-tuning regularization
    'clf__C': [5.0, 10.0, 50.0, 100.0, 200.0],
    
    # Tune the regularization penalty
    'clf__penalty': ['l1', 'l2'] 
}

print(f"Starting Grid Search (testing {3 * 5 * 2} combinations with cross-validation)...")

grid_search = GridSearchCV(
    text_classifier_pipeline, 
    param_grid, 
    cv=5, 
    scoring='f1_macro', 
    verbose=2, 
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Grid Search complete.")
print(f"Best parameters found: {grid_search.best_params_}")


# --------------------------------------------------------------- 
# Evaluate the Best Model
# --------------------------------------------------------------- 
best_pipeline = grid_search.best_estimator_

y_pred = best_pipeline.predict(X_test)

print("\n--- Model Evaluation Report ---")
print(classification_report(y_test, y_pred, zero_division=0))


# --------------------------------------------------------------- 
# Save the Trained Pipeline 
# --------------------------------------------------------------- 
model_filename = 'transactions_categorizer_model.pkl'

with open(model_filename, 'wb') as file:
    pickle.dump(best_pipeline, file)

print(f"\nModel saved successfully as {model_filename}")
