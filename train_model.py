import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

print("Loading data...")
data = pd.read_csv("churn_data.csv")

print("Encoding...")
le = LabelEncoder()
cols = ['Gender','ContractType','InternetUsage','PaymentMethod','ServiceType','Churn']

for col in cols:
    data[col] = le.fit_transform(data[col])

X = data.drop('Churn', axis=1)
y = data['Churn']

print("Splitting...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Training model...")
model = RandomForestClassifier(n_estimators=10)  # faster
model.fit(X_train, y_train)

print("Saving model...")
with open('churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Done!")