
 🛡️ Online Payment Fraud Detection

A Python-based machine learning project to detect fraudulent online payment transactions. This project uses **Random Forest Classifier** to predict the probability of fraud based on transaction features and provides a **Streamlit web app** for interactive predictions.

---

 🔹 Project Overview

Fraudulent transactions are a major problem for financial institutions and e-commerce platforms. Detecting fraud in real-time helps reduce financial loss and protect users.  

This project allows users to:

- Input transaction details via a web interface.
- Predict the probability of a transaction being fraudulent.
- Visualize fraud risk in a simple and interactive way.

**Key Features:**

- Machine Learning Model: Random Forest Classifier  
- Interactive Web App: Streamlit  
- Categorical Feature Encoding using `LabelEncoder`  
- Probability-based fraud prediction  

---

 📊 Dataset

The dataset contains historical online payment transactions with the following features:

| Feature Name         | Description                                      
| `amount`            | Transaction amount in INR                        |
| `transaction_type`  | Online / Offline transaction                     |
| `account_age`       | Account age in years                             |
| `customer_location` | City of the customer                              |
| `merchant_category` | Category of merchant (Electronics, Groceries, etc.) |
| `is_fraud`          | Target variable (0 = Legitimate, 1 = Fraud)    |

**Dataset file name:** `transactions.csv`

---

 🧩 Workflow

1. **Data Loading**: Load the CSV dataset (`transactions.csv`) and inspect it.  
2. **Data Preprocessing**:  
   - Handle missing values (if any)  
   - Encode categorical variables (`LabelEncoder`)  
3. **Model Training**:  
   - Split data into training and testing sets  
   - Train a Random Forest Classifier with GridSearchCV  
   - Save the trained model (`fraud_model.pkl`) and encoders (`label_encoders.pkl`)  
4. **Streamlit App**:  
   - Input transaction details  
   - Encode input features using saved label encoders  
   - Predict fraud probability and display results  

---

🚀 Running the Streamlit App

1. Make sure Python and required packages are installed:
```bash
pip install -r requirements.txt