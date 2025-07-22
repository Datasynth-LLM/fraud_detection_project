# 💳 Fraud Transaction Detection System

A Streamlit-powered machine learning web app that predicts whether a financial transaction is **fraudulent** or **legitimate**, based on historical patterns. The system also allows batch predictions, visual insights, and downloadable results — ideal for dashboards, analysts, or fraud control teams.

---

## 🔐 Futuristic Security Theme
The UI and visual elements are designed to feel modern, responsive, and secure — like a live terminal dashboard for transaction fraud detection.

---

## 📁 Project Structure

```
fraud_detection_project/
│
├── app.py                            # Streamlit web app
├── main.py                           # Model training script
├── save_model_test.py                # Model testing utility
├── fraud_project_launcher.bat        # Unified launcher (main + app)
│
├── data/                             # Dataset folder (.pkl files)
│   └── transactions.pkl
│
├── models/                           # Trained model
│   └── fraud_rf_model.pkl
│
├── outputs/                          # Visual EDA results
│   ├── amount_distribution.png
│   ├── confusion_matrix.png
│   ├── fraud_vs_nonfraud.png
│   ├── fraud_over_time.png
│   └── fraud_by_scenario.png
│
├── scripts/                          # Custom Python modules
│   ├── eda.py
│   ├── feature_engineering.py
│   └── load_data.py
│
├── notebooks/                        # Jupyter analysis
│   ├── eda_fraud_analysis.ipynb
│   └── dataframe_exploration.ipynb
│
└── README.md                         # Project documentation
```

---

## 🚀 How to Run the Project

### 🛠️ 1. Set Up Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate       # On Windows
pip install -r requirements.txt
```

### 🧠 2. Train the Model
This step processes the transaction data, generates features, performs EDA, and saves the trained model.
```bash
python main.py
```

### 🌐 3. Launch the Streamlit App
```bash
streamlit run app.py
```
Or simply double-click the shortcut:
```bash
fraud_project_launcher.bat
```

---

## 📦 Features

✅ Predict single transactions based on inputs  
✅ Upload CSV for batch predictions  
✅ View fraud prediction table + download results  
✅ Visualize fraud probability distribution  
✅ Auto-clear form after submission  
✅ Real-time clock & alert system (prob > 90%)  
✅ Light/Dark theme toggle  

---

## 📥 Example Input CSV Format

| TX_DATETIME         | TX_AMOUNT |
|---------------------|-----------|
| 2025-07-22 12:30:00 | 199.00    |
| 2025-07-22 15:45:00 | 520.00    |

---

## 📌 Notes

- Make sure `fraud_rf_model.pkl` is generated before using the app.
- Only `TX_DATETIME` and `TX_AMOUNT` are required in the input CSV.

---

## 👨‍💻 Author

**Pixel Rider** – Data Scientist & ML Developer  
Project powered by OpenAI GPT-4 & Streamlit

---

## 🛡️ License

This project is intended for educational/demo purposes. Not production certified for financial risk systems.