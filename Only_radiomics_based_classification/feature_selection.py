import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn_genetic import GAFeatureSelectionCV
from sklearn_genetic.plots import plot_fitness_evolution

from collections import Counter

# --- Load Data ---
data = pd.read_csv('Only_radiomics_based_classification/radiomics_features_washu2_p1_143_with_labels.csv') 
X = data.drop(columns=['PatientID', 'GT', 'ImagePath', 'Patient ID'])
y = data['GT']

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# --- Define multiple estimators ---
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(kernel='linear', probability=True)
}

selected_features_per_model = {}

# --- Run GA Feature Selection for each model ---
for name, model in models.items():
    print(f"\n🧬 Running GA Feature Selection with {name}")
    selector = GAFeatureSelectionCV(
        estimator=model,
        cv=5,
        scoring="accuracy",
        population_size=30,
        generations=20,
        n_jobs=-1,
        verbose=False,
        keep_top_k=5,
        crossover_probability=0.8,
        mutation_probability=0.2,
        elitism=True,
        algorithm='eaMuPlusLambda'
    )
    
    selector.fit(X_train, y_train)
    selected = X_train.columns[selector.support_].tolist()
    selected_features_per_model[name] = selected

    print(f"✅ {name} selected features ({len(selected)}): {selected}")
    print(f"   Best CV Accuracy: {selector.best_score:.4f}")

# --- Combine feature selections ---
all_selected = [f for features in selected_features_per_model.values() for f in features]
feature_votes = Counter(all_selected)

print("\n🗳️ Feature vote count across models:")
for feat, count in feature_votes.most_common():
    print(f"  {feat}: {count} votes")

# --- Final selection based on vote threshold (e.g., at least 2 models agree) ---
consensus_features = [feat for feat, count in feature_votes.items() if count >= 2]

print("\n✅ Final consensus features selected by at least 2 models:")
print(consensus_features)
