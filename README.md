# 🚀 SpaceX Rocket Landing Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-SciKit--Learn-orange) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Project Overview
This project aims to predict whether a SpaceX Falcon 9 rocket will successfully land after launch. Using historical launch data and machine learning techniques, we build a model to analyze the factors influencing successful landings.

📢 **Disclaimer:** This project is inspired by IBM's work on SpaceX landing predictions. It is intended for educational and research purposes only and does not claim ownership over any proprietary IBM content.

## 📂 Table of Contents
- [📌 Project Overview](#-project-overview)
- [🧪 Features](#-features)
- [🛠️ Tech Stack](#-tech-stack)
- [📊 Dataset](#-dataset)
- [🚀 Model Training](#-model-training)
- [📈 Results](#-results)
- [📖 Usage](#-usage)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

## 🧪 Features
- Data preprocessing and exploratory data analysis (EDA) of SpaceX launches.
- Feature engineering for predictive modeling.
- Machine learning model comparison and evaluation.
- Model deployment and visualization of predictions.

## 🛠️ Tech Stack
- **Programming Language:** Python 🐍
- **Libraries & Frameworks:** Pandas, NumPy, SciKit-Learn, Matplotlib, Seaborn, XGBoost
- **Data Visualization:** Plotly, Folium
- **Model Training:** Logistic Regression, Decision Trees, SVM, KNN

## 📊 Dataset
The dataset includes details of past SpaceX launches, including:
- Launch site
- Payload mass
- Booster version
- Weather conditions
- Landing outcome (successful or failed)

📥 **Source:** Data collection was conducted through **web scraping** and **APIs** from various sources, including [SpaceX API & Historical Data](https://www.kaggle.com/datasets/scoleman/spacex-launch-data).

## 🚀 Model Evaluation
### Models Evaluated:
- **Logistic Regression** ✅ - Best balance of performance, interpretability, and efficiency.
- **Support Vector Machine (SVM)** ❌ - High training cost with no significant performance gain.
- **Decision Tree** ❌ - Prone to overfitting and lacks reliable probability estimation.
- **KNN** ❌ - Lower performance and computationally inefficient for larger datasets.

🔹 **Final Recommendation:** **Logistic Regression** due to its efficiency, interpretability, and well-calibrated probability outputs.

## 📈 Results
- **Best Model:** Logistic Regression (Accuracy: 83%, LogLoss: 0.3)
- **Key Findings:** Rocket landings are strongly influenced by payload mass and booster version.
- **Future Enhancements:** Implementing ensemble methods like Random Forests and Gradient Boosting.

## 📖 Usage
Currently, this project focuses on model evaluation rather than training and deployment. The evaluation process compared multiple models to determine the best-performing one for predicting SpaceX rocket landings.

### 🔹 How to Use the Analysis
- Review the dataset and feature engineering steps in the Jupyter Notebooks.
- Understand the performance of different models and the reasoning behind selecting Logistic Regression.
- Extend the work by implementing training and prediction scripts if needed.### 🔹 Clone Repository
```bash
git clone https://github.com/ChiwamaMM/spacex-landing-prediction.git
cd spacex-landing-prediction
```

### 🔹 Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly xgboost folium
```



## 🤝 Contributing
Contributions are welcome! Please submit a pull request or open an issue if you’d like to improve this project.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

