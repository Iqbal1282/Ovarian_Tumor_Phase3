import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn_genetic import GAFeatureSelectionCV

# Load your CSV
data = pd.read_csv('radiomics_analysis/radiomics_features.csv') 

# Separate features and labels
X = data.drop(columns=['PatientID', 'Label'])  # Assuming your CSV has these two columns
y = data['Label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Define the model
model = RandomForestClassifier(random_state=42)

# Define the Genetic Algorithm Feature Selector
selector = GAFeatureSelectionCV(
    estimator=model,
    cv=5,
    scoring="accuracy",
    population_size=30,
    generations=20,
    n_jobs=-1,
    verbose=True,
    keep_top_k=5,
    crossover_probability=0.8,
    mutation_probability=0.2,
    elitism=True,
    algorithm='eaMuPlusLambda'
)

# Fit
selector.fit(X_train, y_train)

# Best selected features
selected_features = X_train.columns[selector.support_]
print("\n✅ Selected Features:")
print(selected_features.tolist())

# Best score
print(f"\n✅ Best cross-validation accuracy: {selector.best_score_:.4f}")
