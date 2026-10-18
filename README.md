# AI Complaint Management System

An AI-powered web application that automatically classifies college complaints, estimates their priority, and helps administrators manage complaint resolution.

## Features

- **AI Complaint Classification:** Categorizes complaints into Electrical, Water & Plumbing, Internet, Cleaning, Food, Security, and General.
- **Priority Detection:** Assigns High, Medium, or Low priority using predefined keyword rules.
- **Confidence Score:** Displays the model's highest predicted class probability.
- **Complaint Tracking:** Allows users to track submitted complaints using their complaint ID.
- **Admin Dashboard:** Displays complaints, categories, priorities, and statuses.
- **Status Management:** Supports Pending, In Progress, and Resolved statuses.
- **Database:** Stores complaint records using SQLite.

## Technologies Used

- Python
- Flask
- Scikit-learn
- TF-IDF (word and character features)
- Logistic Regression
- SQLite
- HTML and CSS

## Machine Learning Evaluation

The current dataset contains 255 unique complaints across seven categories.

- **Held-out test accuracy:** 98.04% (51 test examples)
- **5-fold cross-validation accuracy:** 93.73%
- **Model:** Logistic Regression with combined word-level and character-level TF-IDF features

These results are estimates based on the current dataset and do not guarantee the same accuracy on new real-world complaints.


   python app.py
   ```
## Run Locally

### 1. Prerequisites
- Python 3.10 or later
- Git (optional, for cloning the repository)

### 2. Clone the Repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AI-Complaint-Management-System
```
Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

Alternatively, download the repository as a ZIP file and extract it.

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root using `.env.example` as a template. Set your own secret key, admin username, and strong admin password.

Never upload your actual `.env` file to GitHub.

### 5. Run the Application
```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

The SQLite database is initialized when the application starts.

## Project Structure

```text
AI_Complaint_Management_System/
├── app.py
├── train_model.py
├── evaluate_model_confidence.py
├── complaints_dataset.csv
├── complaint_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
├── README.md
├── templates/
└── static/
```

## Future Improvements

- Improve classification using a larger and more diverse dataset.
- Improve priority detection and duplicate complaint handling.
- Add secure administrator authentication.
- Improve confidence calibration and evaluate more unseen complaints.
- Deploy the application online.

## Disclaimer

This project is an academic prototype. AI predictions and keyword-based priority assignments should be reviewed before being used for important decisions.
