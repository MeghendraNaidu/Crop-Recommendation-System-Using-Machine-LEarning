import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
crop = pd.read_csv(r"C:\Users\Meghendra\Desktop\Major Project\Crop Recommendation\Crop_recommendation.csv")

# Check initial data
print(crop.head())
print(crop.shape)
print(crop.info())

# Handle missing values and duplicates
print(crop.isnull().sum())
print(crop.duplicated().sum())

# Encode categorical target variable
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 
    'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10, 
    'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15, 
    'blackgram': 16, 'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19, 
    'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}

crop['label'] = crop['label'].map(crop_dict)  # Convert labels to numerical values

# Compute and visualize correlation matrix
print(crop.corr())  # Now it works correctly
sns.heatmap(crop.corr(), annot=True, cbar=True)
plt.show()

# Visualize histogram of features
sns.histplot(crop['P'])
plt.show()

sns.histplot(crop['N'])
plt.show()

# Prepare dataset for training
X = crop.drop('label', axis=1)
y = crop['label']

# Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler
mx = MinMaxScaler()
sc = StandardScaler()

X_train = mx.fit_transform(X_train)
X_test = mx.transform(X_test)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Model training
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

models = {
    'LogisticRegression': LogisticRegression(),
    'GaussianNB': GaussianNB(),
    'SVC': SVC(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'ExtraTreeClassifier': ExtraTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'BaggingClassifier': BaggingClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"{name} model accuracy: {score}")

# Train final Random Forest Classifier
randclf = RandomForestClassifier()
randclf.fit(X_train, y_train)

y_pred = randclf.predict(X_test)
print(f"RandomForest Accuracy: {accuracy_score(y_test, y_pred)}")

# Crop Recommendation Function
def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features = mx.transform(features)  # Use transform, not fit_transform
    features = sc.transform(features)
    prediction = randclf.predict(features)
    return prediction[0]

# Test Recommendation Function
N, P, K, temperature, humidity, ph, rainfall = 90, 42, 43, 20.88, 82.00, 6.50, 202.94
predict = recommendation(N, P, K, temperature, humidity, ph, rainfall)
print(f"Predicted crop: {predict}")

# Save Model
import pickle
pickle.dump(randclf, open('model.pkl', 'wb'))
pickle.dump(mx, open('minmaxscaler.pkl', 'wb'))
pickle.dump(sc, open('standscaler.pkl', 'wb'))

test_samples = [
    [33, 23, 45, 20.00, 85.83, 7.11, 112.33],  # Example 1
    [15, 77, 20, 25.13, 66.92, 7.39, 49.04],  # Example 2
    [44, 75, 22, 30.03, 64.14, 7.57, 71.21],  # Example 3
]

for sample in test_samples:
    pred = recommendation(*sample)
    print(f"Input: {sample} => Predicted Crop ID: {pred}")

