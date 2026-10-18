
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion


# 1. Load dataset
data = pd.read_csv("complaints_dataset.csv")

print("Dataset loaded successfully!")
print("Original number of complaints:", len(data))

# Check required columns and missing values
required_columns = {"complaint", "category"}

if not required_columns.issubset(data.columns):
    raise ValueError("CSV must contain 'complaint' and 'category' columns.")

data = data.dropna(subset=["complaint", "category"]).copy()

# Normalize text for duplicate detection
data["clean_text"] = (
    data["complaint"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.split()
    .str.join(" ")
)

# Remove empty complaints
data = data[data["clean_text"] != ""].copy()

# Check whether identical complaints have different labels
label_counts = data.groupby("clean_text")["category"].nunique()
conflicts = label_counts[label_counts > 1]

if not conflicts.empty:
    print("\nERROR: Identical complaint texts have different categories:")
    print(
        data[data["clean_text"].isin(conflicts.index)]
        [["complaint", "category"]]
        .to_string(index=False)
    )
    raise ValueError(
        "Fix conflicting category labels before training."
    )

# Remove duplicates after normalization
duplicate_count = data.duplicated(subset=["clean_text"]).sum()
print("Duplicate complaints removed:", duplicate_count)

data = data.drop_duplicates(
    subset=["clean_text"]
).reset_index(drop=True)

print("Dataset size after cleaning:", len(data))
print("\nComplaints per category:")
print(data["category"].value_counts())

# 2. Separate input and output
X = data["complaint"]
y = data["category"]

# Ensure each category has enough examples for stratified evaluation
if y.value_counts().min() < 5:
    raise ValueError(
        "At least 5 examples per category are needed for 5-fold "
        "stratified cross-validation."
    )

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Diagnostic: check whether the target complaint is in training data
target = "The classroom has a lot of dust"

target_normalized = " ".join(target.strip().lower().split())

normalized_train = (
    X_train.astype(str)
    .str.strip()
    .str.lower()
    .str.split()
    .str.join(" ")
)

normalized_test = (
    X_test.astype(str)
    .str.strip()
    .str.lower()
    .str.split()
    .str.join(" ")
)

print("\nTARGET COMPLAINT CHECK")
print("Present in full dataset:", target_normalized in set(
    data["clean_text"]
))
print("Present in training data:", target_normalized in set(
    normalized_train
))
print("Present in test data:", target_normalized in set(
    normalized_test
))



# 4. Convert text into combined word + character TF-IDF features
vectorizer = FeatureUnion([
    (
        "word_tfidf",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            strip_accents="unicode",
            sublinear_tf=True
        )
    ),
    (
        "char_tfidf",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            max_features=30000
        )
    )
])

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Create and train the model
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)


# 5. Create and train the model
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)

# 6. Evaluate on the held-out test set
predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)

print("\nModel trained successfully!")
print(f"Test Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# 6A. Evaluate predicted probabilities and confidence calibration
test_probabilities = model.predict_proba(X_test_tfidf)

# Highest predicted probability for each complaint
test_confidences = test_probabilities.max(axis=1)

# Whether each prediction was correct
test_correct = (
    predictions == y_test.to_numpy()
)

# Log loss: lower is generally better
test_log_loss = log_loss(
    y_test,
    test_probabilities,
    labels=model.classes_
)

print("\nConfidence Evaluation:")
print(f"Test Log Loss: {test_log_loss:.4f}")
print(
    f"Average Model Confidence: "
    f"{test_confidences.mean() * 100:.2f}%"
)
print(
    f"Accuracy on Test Set: "
    f"{test_correct.mean() * 100:.2f}%"
)

# Compare confidence with actual accuracy in confidence bands
print("\nConfidence Calibration by Band:")
print("Confidence Band | Samples | Avg Confidence | Actual Accuracy")

confidence_bands = [
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.01)
]

for lower, upper in confidence_bands:
    mask = (
        (test_confidences >= lower)
        & (test_confidences < upper)
    )

    count = int(mask.sum())

    if count > 0:
        avg_confidence = test_confidences[mask].mean()
        actual_accuracy = test_correct[mask].mean()

        print(
            f"{lower:.0%}-{min(upper, 1.0):.0%}          "
            f"| {count:7d} "
            f"| {avg_confidence * 100:12.2f}% "
            f"| {actual_accuracy * 100:14.2f}%"
        )
    else:
        print(
            f"{lower:.0%}-{min(upper, 1.0):.0%}          "
            f"|       0 | No samples"
        )


# Show incorrect predictions
print("\nWrong Predictions:")

wrong_count = 0

for complaint, actual, predicted in zip(
    X_test, y_test, predictions
):
    if actual != predicted:
        wrong_count += 1
        print("Complaint:", complaint)
        print("Actual:", actual)
        print("Predicted:", predicted)
        print()

if wrong_count == 0:
    print("No incorrect predictions in this test split.")


# 7. Cross-validation using the combined word + character model
cv_pipeline = Pipeline([
    (
        "features",
        FeatureUnion([
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    strip_accents="unicode",
                    sublinear_tf=True
                )
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                    max_features=30000
                )
            )
        ])
    ),
    (
        "model",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    cv_pipeline,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("\n5-Fold Cross-Validation Scores:")
print([round(score, 4) for score in cv_scores])
print(f"Average CV Accuracy: {cv_scores.mean():.2%}")
print(f"CV Standard Deviation: {cv_scores.std():.2%}")




# 8. Train the final combined model on all cleaned data
final_vectorizer = FeatureUnion([
    (
        "word_tfidf",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            strip_accents="unicode",
            sublinear_tf=True
        )
    ),
    (
        "char_tfidf",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            max_features=30000
        )
    )
])

X_all_tfidf = final_vectorizer.fit_transform(X)

final_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

final_model.fit(X_all_tfidf, y)

# Save model and matching feature extractor
joblib.dump(final_model, "complaint_model.pkl")
joblib.dump(final_vectorizer, "tfidf_vectorizer.pkl")

# Use the final model for the prediction tests below
model = final_model
vectorizer = final_vectorizer

print("\nFinal combined model trained on all cleaned complaints.")
print("Model and vectorizer saved successfully!")


# 9. Test new complaints and show confidence
test_complaints = [
    "The ceiling fan in my classroom has stopped working",
    "The hostel bathroom has no water",
    "My hostel WiFi keeps disconnecting",
    "The classroom floor is covered with dirt",
    "Someone stole my laptop from the hostel",
    "The mess served cold food today",
    "The classroom projector is not working",
    "The washroom tap is leaking",
    "The electrical wiring in the laboratory is damaged",
    "There is garbage near the classroom"
]

test_vectors = vectorizer.transform(test_complaints)

predicted_categories = model.predict(test_vectors)
probabilities = model.predict_proba(test_vectors)
confidences = probabilities.max(axis=1) * 100

print("\nNew Complaint Predictions:")

for complaint, category, confidence in zip(
    test_complaints,
    predicted_categories,
    confidences
):
    print(f"\nComplaint: {complaint}")
    print(f"Predicted Category: {category}")
    print(f"Model Confidence: {confidence:.2f}%")
