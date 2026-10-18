
import pandas as pd
import joblib

model = joblib.load("complaint_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# New wording for diagnostic testing
test_cases = [
    ("Fine dust has settled on every classroom table", "Cleaning"),
    ("The rubbish outside our hostel has not been removed", "Cleaning"),

    ("The laboratory sockets have stopped supplying power", "Electrical"),
    ("The classroom lights keep going on and off", "Electrical"),

    ("The cafeteria meal was undercooked today", "Food"),
    ("The drinking water tastes strange in the mess", "Food"),

    ("The classroom projector shows a blank screen", "General"),
    ("A broken chair in the lecture hall needs replacement", "General"),

    ("My laptop cannot connect to the campus network", "Internet"),
    ("The campus WiFi disconnects every few minutes", "Internet"),

    ("Someone entered our room and took my bag", "Security"),
    ("The hostel gate was left unlocked overnight", "Security"),

    ("The bathroom drain is overflowing with water", "Water & Plumbing"),
    ("No water comes out when I open the kitchen tap", "Water & Plumbing"),
]

# Exclude exact complaints already in the dataset
data = pd.read_csv("complaints_dataset.csv")

existing = set(
    data["complaint"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.lower()
    .str.split()
    .str.join(" ")
)

unseen_cases = [
    (complaint, expected)
    for complaint, expected in test_cases
    if " ".join(complaint.strip().lower().split()) not in existing
]

if not unseen_cases:
    print("No unseen exact complaints available.")
    raise SystemExit

complaints = [item[0] for item in unseen_cases]
expected = [item[1] for item in unseen_cases]

X = vectorizer.transform(complaints)
predictions = model.predict(X)
probabilities = model.predict_proba(X)

correct = 0

print("\nUNSEEN COMPLAINT TEST\n")

for complaint, actual, predicted, probs in zip(
    complaints, expected, predictions, probabilities
):
    confidence = probs.max() * 100
    is_correct = actual == predicted
    correct += int(is_correct)

    print("Complaint:", complaint)
    print("Expected:", actual)
    print("Predicted:", predicted)
    print(f"Confidence: {confidence:.2f}%")
    print("Result:", "CORRECT" if is_correct else "INCORRECT")
    print("-" * 50)

print(f"\nCorrect: {correct}/{len(unseen_cases)}")
print(f"Diagnostic accuracy: {correct / len(unseen_cases) * 100:.2f}%")
print("This is a small diagnostic test, not a definitive accuracy estimate.")
