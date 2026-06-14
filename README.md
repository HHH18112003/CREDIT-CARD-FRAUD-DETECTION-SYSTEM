# Credit Card Fraud Detection System using Advanced Machine Learning

An end-to-end Machine Learning solution designed to detect and classify fraudulent credit card transactions. This repository covers the entire data science pipeline: from exploratory data analysis (EDA) and handling highly imbalanced transaction data, to multi-model training, evaluation, and deploying a functional real-time prediction local API using FastAPI and Ngrok.

## 📌 Key Features
- **Comprehensive EDA:** Detailed visualization of class distribution, transaction behavior, and feature correlations using Matplotlib and Seaborn.
- **Robust Preprocessing:** Automated data cleaning, null handling, and feature scaling using Scikit-Learn's `StandardScaler` to optimize model convergence.
- **Multi-Model Benchmark:** Comparative evaluation of 6 distinct classification algorithms: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors (KNN), Support Vector Classifier (SVC), and Multi-Layer Perceptron (MLP).
- **Production-Ready Deployment:** Best model serialization with `pickle` and exposed as a lightweight REST API endpoint using **FastAPI**.
- **Public Tunneling:** Integrated with **Ngrok** to expose the local API server safely for external client testing.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn
- **API Deployment:** FastAPI, Uvicorn, Ngrok
- **Model Storage:** Pickle

## 📊 Evaluation & Results
The models were trained and cross-validated on a massive Kaggle dataset containing over 560,000 transaction entries. Performance was meticulously measured using standard classification metrics:
- **Metrics Evaluated:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Cross-Validation Score.
- **Production Selection:** The optimal model architecture was selected based on its ability to minimize financial risks (maximizing Recall for fraud detection while maintaining high Precision).

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/HHH18112003/Credit-Card-Fraud-Detection.git](https://github.com/HHH18112003/Credit-Card-Fraud-Detection.git)
   cd Credit-Card-Fraud-Detection
1. Install dependencies:
   pip install numpy pandas matplotlib seaborn scikit-learn fastapi uvicorn requests flask-ngrok nest_asyncio
2. Run the FastAPI Server with Ngrok:
   Execute the deployment cell or run your script. The application will initialize nest_asyncio and flask_ngrok to provide a public URL:
   import uvicorn
# Server will stream predictions at host 0.0.0.0:8000 masked behind a public Ngrok tunnel
3. Test Predictions via Client:
   Send a POST request containing transaction feature inputs to the endpoint NGROK_URL/predict/{model_name} to retrieve real-time fraud alerts (Label 1: Fraud, Label 0: Real).
