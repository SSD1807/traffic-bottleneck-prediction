# 🚦 Traffic Bottleneck Prediction & Smart Routing System

An intelligent traffic analysis system that predicts congestion using Machine Learning and dynamically recommends optimal routes using graph-based algorithms.

---

## 🔥 Key Features

* 📊 **ML-Based Traffic Prediction**
  Predicts congestion levels (Low, Medium, High) using trained models (XGBoost / Random Forest)

* 🧠 **Smart Route Optimization**
  Uses graph algorithms (Dijkstra) to compute optimal routes

* 🚨 **Bottleneck Detection**
  Identifies high-congestion nodes and critical traffic zones

* 🔄 **Dynamic Rerouting**
  Avoids congested roads and suggests alternative paths in real-time

* 📍 **Interactive Dashboard (Streamlit)**
  Displays:

  * Traffic conditions
  * Route comparisons
  * Network visualization

---

## 🧠 Tech Stack

* **Programming Language:** Python
* **Machine Learning:** Scikit-learn, XGBoost
* **Graph Algorithms:** NetworkX
* **Visualization:** Streamlit, Matplotlib
* **Data Processing:** Pandas, NumPy

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

---

## ⚙️ Project Workflow

1. **Data Preprocessing**

```bash
python phase1_preprocess.py
```

2. **Model Training**

```bash
python phase2_train.py
```

3. **Run Dashboard**

```bash
streamlit run dashboard.py
```

---

## 🧩 System Architecture

* Raw Traffic Data → Feature Engineering
* ML Model → Predict congestion levels
* Graph Engine → Apply dynamic weights
* Routing Algorithm → Compute optimal paths
* Dashboard → Visualize traffic and routes

---

## 📊 Dataset

Datasets are not included due to size constraints.

To regenerate datasets:

```bash
python generate_bhubaneswar_traffic.py
python merge_and_retrain.py
```

---

## 🎯 Project Objective

To design a scalable system that:

* Predicts urban traffic congestion
* Detects bottleneck regions
* Suggests optimal routes dynamically
* Simulates real-world traffic conditions

---

## 📸 Demo

Add your dashboard screenshot here:

```markdown
![Dashboard](screenshot.png)
```

---

## 👨‍💻 Authors

* Shreeyan Satwik Das
* Sambit Kumar Sahoo
* Sanat Kumar
* Sakti Sourav Das

---

## 🚀 Future Enhancements

* Integration with real-time traffic APIs (Google Maps / OpenTraffic)
* Implementation of A* algorithm for faster routing
* Live GPS-based traffic input
* Deployment as a full web application

---