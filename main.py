import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = pd.read_csv("House_Rent_Dataset.csv")

df = pd.DataFrame(data)

numeric_values = ['BHK', 'Size', 'Bathroom']
categorical_values = ['Point of Contact', 'Floor', 'City', 'Furnishing Status']

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_values),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_values)
    ]
)
model = Pipeline(steps=[("preprocessor", preprocessor),
                        ("regressor", DecisionTreeRegressor(
                            random_state=42,
                            max_depth=10,
                            min_samples_split=20,
                            min_samples_leaf=14
                        ))]
                 )

X = df[numeric_values + categorical_values]
y = df["Rent"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"RMSE : {rmse}")
print("R_Squared : ", r2_score(y_test, y_pred))
