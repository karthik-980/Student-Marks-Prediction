import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("dataset.csv")

# Input and Output
X = data[["hours"]]
y = data["marks"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
score = model.score(X_test, y_test)

print("Model Accuracy:", score)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully")