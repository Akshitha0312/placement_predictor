import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

data = pd.read_csv("placement.csv")

X = data.drop("placed", axis=1)
y = data["placed"]

model = LogisticRegression()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
print("Model trained and saved!")
