
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    classification_report
)

# 1. Load and clean the existing dataset
data = pd.read_csv("complaints_dataset.csv")
data = data.dropna(subset=["complaint", "category"]).copy()

data["clean_text"] = (
    data["complaint"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.split()
    .str.join(" ")
)

data = data[data["clean_text"] != ""].copy()

# Stop if the same complaint has conflicting labels
conflicts = data.groupby("clean_text")["category"].nunique()
if (conflicts > 1).any():
    raise ValueError(
        "Identical complaints have different labels. "
        "Resolve conflicting labels before evaluation."
    )

data = data.drop_duplicates(
    subset=["clean_text"]
).reset_index(drop=True)

X = data["complaint"]
y = data["category"]
labels = np.sort(y.unique())

print("Unique complaints:", len(data))
print("Categories:", list(labels))
print("\nCategory counts:")
print(y.value_counts().to_string())

# 2. Use the same folds for both models
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Model A: your current word-based model
word_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        strip_accents="unicode"
    )),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

# Model B: combine word patterns and character patterns
combined_model = Pipeline([
    ("features", FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            strip_accents="unicode",
            sublinear_tf=True
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            max_features=30000
        ))
    ])),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

models = [
    ("Current word-based model", word_model),
    ("Combined word + character model", combined_model)
]

results = {}

# 3. Evaluate both models using out-of-fold predictions
for name, pipeline in models:
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    predictions = cross_val_predict(
        pipeline, X, y, cv=cv, method="predict"
    )

    probabilities = cross_val_predict(
        pipeline, X, y, cv=cv, method="predict_proba"
    )

    accuracy = accuracy_score(y, predictions)
    loss = log_loss(y, probabilities, labels=labels)

    results[name] = {
        "accuracy": accuracy,
        "log_loss": loss
    }

    print(f"Out-of-fold accuracy: {accuracy:.2%}")
    print(f"Log loss: {loss:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y,
        predictions,
        labels=labels,
        zero_division=0
    ))

    wrong = np.where(predictions != y.to_numpy())[0]

    print("Incorrect predictions:", len(wrong))
    for i in wrong:
        print(
            f"Actual: {y.iloc[i]} | "
            f"Predicted: {predictions[i]} | "
            f"Complaint: {X.iloc[i]}"
        )

# 4. Compare results
print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

for name, result in results.items():
    print(
        f"{name}: "
        f"Accuracy={result['accuracy']:.2%}, "
        f"Log loss={result['log_loss']:.4f}"
    )

print("\nBaseline accuracy: 90.98%")
print("Baseline log loss: 1.1037")
print("Choose based on accuracy, log loss, and error patterns.")
