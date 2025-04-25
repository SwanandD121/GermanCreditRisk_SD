# credit_risk.py - German Credit Risk Prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

def load_and_prepare_data():
    """
    Load the German Credit dataset and prepare it for analysis.
    The UCI German Credit dataset has a specific format with 20 attributes and a binary target.
    """
    # Load dataset - adjust path if needed
    try:
        # Try the format that matches the kaggle dataset linked
        data = pd.read_csv("german_credit_data.csv")
        print("Loaded german_credit_data.csv successfully")
        
        # If this is the newer format, the target might not be included
        # We'll check for this and handle accordingly
        if 'Risk' not in data.columns and 'Class' not in data.columns and 'Target' not in data.columns:
            print("Target variable not found - attempting to load original UCI format")
            raise FileNotFoundError
            
    except FileNotFoundError:
        # Try the original UCI format (space-separated, no header)
        try:
            # The original UCI format has 20 attributes plus the target
            column_names = [
                'status', 'duration', 'credit_history', 'purpose', 'amount',
                'savings', 'employment_duration', 'installment_rate', 'personal_status_sex',
                'other_debtors', 'present_residence', 'property', 'age', 'other_installment_plans',
                'housing', 'existing_credits', 'job', 'people_liable', 'telephone', 'foreign_worker',
                'target'  # 1 = good credit, 2 = bad credit
            ]
            data = pd.read_csv("german.data", sep=' ', header=None, names=column_names)
            print("Loaded german.data successfully")
        except FileNotFoundError:
            # Provide a direct URL as a fallback
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
            print(f"Local files not found, attempting to load from: {url}")
            column_names = [
                'status', 'duration', 'credit_history', 'purpose', 'amount',
                'savings', 'employment_duration', 'installment_rate', 'personal_status_sex',
                'other_debtors', 'present_residence', 'property', 'age', 'other_installment_plans',
                'housing', 'existing_credits', 'job', 'people_liable', 'telephone', 'foreign_worker',
                'target'  # 1 = good credit, 2 = bad credit
            ]
            data = pd.read_csv(url, sep=' ', header=None, names=column_names)
    
    # Convert target to binary (0=good, 1=bad) if needed
    # The original dataset uses 1=good, 2=bad
    if 'target' in data.columns and data['target'].isin([1, 2]).all():
        data['target'] = data['target'] - 1  # Convert to 0=good, 1=bad
    
    # For the newer format, we might need different preprocessing
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
    
    # If target still doesn't exist, create a synthetic one 
    # (this is just a fallback and not recommended for real analysis)
    if 'target' not in data.columns:
        print("WARNING: Creating synthetic target variable. This is not recommended for real analysis.")
        data['target'] = np.where(
            ((data['Credit amount'] > 5000) & (data['Duration'] > 24)) |
            ((data['Saving accounts'].fillna('little') == 'little') &
            (data['Checking account'].fillna('little') == 'little')),
            1,  # Bad risk
            0   # Good risk
        )
    
    return data

def exploratory_data_analysis(data):
    """Perform exploratory data analysis and visualization"""
    print("\n=== Exploratory Data Analysis ===")
    print(f"Dataset shape: {data.shape}")
    print("\nFirst 5 rows:")
    print(data.head())
    
    print("\nData types:")
    print(data.dtypes)
    
    print("\nSummary statistics:")
    print(data.describe(include='all').T)
    
    print("\nMissing values:")
    print(data.isnull().sum())
    
    # Check target distribution
    target_col = 'target' if 'target' in data.columns else 'Credit_risk'
    print(f"\nTarget variable ({target_col}) distribution:")
    target_dist = data[target_col].value_counts(normalize=True)
    print(target_dist)
    
    # Visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Target Distribution
    plt.subplot(2, 2, 1)
    sns.countplot(x=target_col, data=data)
    plt.title(f'Credit Risk Distribution', fontsize=12)
    plt.xlabel('Credit Risk (0=Good, 1=Bad)', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    
    # Determine numerical columns for plotting
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # If 'amount' or 'Credit amount' is in the dataset
    amount_col = 'amount' if 'amount' in numeric_cols else 'Credit amount' if 'Credit amount' in numeric_cols else None
    if amount_col:
        # Plot 2: Credit Amount by Risk
        plt.subplot(2, 2, 2)
        sns.boxplot(x=target_col, y=amount_col, data=data)
        plt.title(f'{amount_col} vs Risk', fontsize=12)
        plt.xlabel('Credit Risk (0=Good, 1=Bad)', fontsize=10)
        plt.ylabel(amount_col, fontsize=10)
    
    # If 'duration' or 'Duration' is in the dataset
    duration_col = 'duration' if 'duration' in numeric_cols else 'Duration' if 'Duration' in numeric_cols else None
    if duration_col:
        # Plot 3: Duration by Risk
        plt.subplot(2, 2, 3)
        sns.boxplot(x=target_col, y=duration_col, data=data)
        plt.title(f'{duration_col} vs Risk', fontsize=12)
        plt.xlabel('Credit Risk (0=Good, 1=Bad)', fontsize=10)
        plt.ylabel(duration_col, fontsize=10)
    
    # If 'age' or 'Age' is in the dataset
    age_col = 'age' if 'age' in numeric_cols else 'Age' if 'Age' in numeric_cols else None
    if age_col:
        # Plot 4: Age by Risk
        plt.subplot(2, 2, 4)
        sns.boxplot(x=target_col, y=age_col, data=data)
        plt.title(f'{age_col} vs Risk', fontsize=12)
        plt.xlabel('Credit Risk (0=Good, 1=Bad)', fontsize=10)
        plt.ylabel(age_col, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('credit_risk_eda.png')
    plt.close()
    
    # Create correlation heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        corr_matrix = data[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                    cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix', fontsize=14)
        plt.savefig('correlation_matrix.png')
        plt.close()
    
    return target_col

def preprocess_data(data, target_col):
    """Preprocess the data for machine learning"""
    
    # Separate features and target
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"\nCategorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    # Create preprocessing pipelines
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, preprocessor, categorical_cols, numerical_cols

def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    """Train and evaluate multiple machine learning models"""
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    best_models = {}
    results = {}
    
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        
        # Create a pipeline with preprocessing and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Define hyperparameters for grid search
        if name == 'Random Forest':
            param_grid = {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2]
            }
        else:  # Gradient Boosting
            param_grid = {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__max_depth': [3, 5],
                'classifier__min_samples_split': [2, 5]
            }
        
        # Perform grid search with cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )
        
        # Fit the grid search
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        best_models[name] = best_model
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_prob)
        
        # Store results
        results[name] = {
            'best_params': grid_search.best_params_,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        # Print results
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    # Find the best overall model
    best_model_name = max(results, key=lambda k: results[k]['roc_auc'])
    print(f"\n=== Best Overall Model: {best_model_name} ===")
    print(f"ROC AUC: {results[best_model_name]['roc_auc']:.4f}")
    
    return best_models, results, best_model_name

def visualize_results(results, X_test, y_test):
    """Visualize model results"""
    plt.figure(figsize=(15, 10))
    
    # Plot ROC curves
    plt.subplot(2, 2, 1)
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
        roc_auc = result['roc_auc']
        plt.plot(fpr, tpr, label=f'{name} (ROC AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    
    # Plot confusion matrices
    for i, (name, result) in enumerate(results.items(), 2):
        plt.subplot(2, 2, i)
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Good', 'Bad'],
                    yticklabels=['Good', 'Bad'])
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('model_results.png')
    plt.close()
    
    return

def analyze_feature_importance(best_models, best_model_name, categorical_cols, numerical_cols, preprocessor):
    """Analyze and visualize feature importance"""
    # Get the best model
    best_model = best_models[best_model_name]
    
    # Check if the model has feature_importances_ attribute
    if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
        # Get feature names from the preprocessor
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = list(ohe.get_feature_names_out(categorical_cols))
        all_features = numerical_cols + cat_features
        
        # Get feature importances
        importances = best_model.named_steps['classifier'].feature_importances_
        
        # Create a dataframe for feature importances
        feature_importance = pd.DataFrame({
            'Feature': all_features,
            'Importance': importances
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title(f'Top 20 Feature Importances - {best_model_name}')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        # Print top 10 features
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
        
        return feature_importance
    else:
        print("\nThe selected model doesn't have feature_importances_ attribute.")
        return None

def main():
    """Main function to run the credit risk prediction pipeline"""
    print("=== German Credit Risk Prediction ===")
    
    # Load and prepare data
    data = load_and_prepare_data()
    
    # Perform exploratory data analysis
    target_col = exploratory_data_analysis(data)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor, categorical_cols, numerical_cols = preprocess_data(data, target_col)
    
    # Train and evaluate models
    best_models, results, best_model_name = train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor)
    
    # Visualize results
    visualize_results(results, X_test, y_test)
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(best_models, best_model_name, categorical_cols, numerical_cols, preprocessor)
    
    print("\n=== Credit Risk Prediction Analysis Completed ===")
    print(f"Best Model: {best_model_name}")
    print(f"ROC AUC: {results[best_model_name]['roc_auc']:.4f}")
    
    # Print business insights
    print("\n=== Business Insights ===")
    print("1. The model identified factors that are most predictive of credit risk.")
    print("2. These insights can help banks make better lending decisions.")
    print("3. Model performance metrics show how reliable these predictions are.")
    print("4. Regular monitoring and updating of the model is recommended.")

if __name__ == "__main__":
    main()