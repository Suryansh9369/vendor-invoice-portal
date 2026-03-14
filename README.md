# 🤖 Vendor Invoice Intelligent Portal

An internal AI-driven analytics portal built with **Streamlit** that leverages 
Machine Learning to streamline freight cost forecasting and vendor invoice 
risk detection.

## 🚀 Features

- 🚚 **Freight Cost Prediction** — Predicts freight cost from invoice quantity 
  and dollar values to support budgeting and vendor negotiations.
- 🚨 **Invoice Manual Approval Flagging** — Detects risky or abnormal vendor 
  invoices that require manual review based on cost and delivery patterns.

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **ML Models:** Scikit-learn (trained & serialized with Joblib/Pickle)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly Express
- **Database:** SQLite (inventory.db)

## 📁 Project Structure
```
├── app.py                        # Main Streamlit app
├── inference/
│   ├── predict_freight.py        # Freight cost inference
│   └── predict_invoice_flag.py   # Invoice flag inference
├── Freight Cost Prediction/
│   ├── train.py
│   ├── data_preprocessing.py
│   └── model_evaluation.py
├── Invoice Flagging/
│   ├── train.py
│   ├── data_preprocessing.py
│   └── model_evaluation.py
├── models/
│   ├── predict_freight_model.pkl
│   ├── predict_flag_invoice.pkl
│   └── scaler.pkl
└── data/
    └── inventory.db
```

## ⚙️ Setup & Installation
```bash
# Clone the repository
git clone https://github.com/your-username/vendor-invoice-portal.git
cd vendor-invoice-portal

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📊 Models

| Model | Input Features | Output |
|---|---|---|
| Freight Cost Prediction | Quantity, Dollars | Estimated Freight Cost ($) |
| Invoice Flag Prediction | Invoice Qty, Invoice $, Freight, Total Item Qty, Total Item $ | Flag / No Flag |

## 💼 Business Impact

- 📈 Improve cost forecasting accuracy
- 📝 Reduce invoice fraud and anomalies
- ⚙️ Faster and smarter finance operations