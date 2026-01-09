# Indian City Rent Prediction Engine

**Live Demo:** [Link to your Streamlit App]

##  Project Overview
Finding fair rental prices in Indian metros is chaotic due to unstandardized listing data. This project is an end-to-end Machine Learning web application that estimates fair market value for residential properties in 5 major Indian cities (Mumbai, Bangalore, Hyderabad, Delhi, Chennai).

Built with **Python**, **Scikit-Learn Pipelines**, and **Streamlit**.

##  Technical Architecture
The system uses a robust `sklearn.pipeline.Pipeline` to ensure training and inference data undergo identical preprocessing, preventing data leakage.

### The Pipeline Steps:
1.  **Data Ingestion:** Loads raw real estate data.
2.  **Preprocessing (`ColumnTransformer`):**
    * **Numeric Features:** Applied `StandardScaler` to `BHK`, `Size`, and `Bathroom` to normalize variance.
    * **Categorical Features:** Applied `OneHotEncoder` (with `handle_unknown='ignore'`) to `City`, `Furnishing Status`, `Floor`, and `Point of Contact`.
3.  **Modeling:**
    * **Algorithm:** `DecisionTreeRegressor`
    * **Hyperparameter Tuning:** Optimized tree structure to prevent overfitting:
        * `max_depth=10` (Limits complexity)
        * `min_samples_leaf=14` (Ensures generalization)
        * `min_samples_split=20` (Prevents isolating outliers)
4.  **Deployment:** Model serialized via `joblib` and served through a Streamlit frontend.

##  Model Performance
* **Metric:** RMSE (Root Mean Squared Error) and R² Score.
* **Performance:** The Decision Tree captures non-linear price jumps (e.g., the sudden spike in rent for "Fully Furnished" vs "Unfurnished") better than linear baselines.
* **Key Insight:** "Size" and "City" were the dominant feature importances, while "Point of Contact" helped identify broker-inflated prices.

##  Tech Stack
* **Core:** Python 3.9+
* **ML Libraries:** Scikit-learn, Pandas, NumPy
* **Preprocessing:** `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`
* **Frontend:** Streamlit

##  How to Run Locally

1. **Clone the repository**
   ```bash
   git clone [https://github.com/AK3D2Y/Rent-Prediction-Model.git](https://github.com/AK3D2Y/Rent-Prediction-Model.git)
