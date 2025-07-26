import numpy as np
import pickle

# Load the trained model and scalers
randclf = pickle.load(open('model.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))

def test_recommendation(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features = mx.transform(features)  # Scale features
    features = sc.transform(features)  # Apply standardization
    prediction = randclf.predict(features)
    return prediction[0]

# Sample Test
N, P, K, temperature, humidity, ph, rainfall = 33, 23, 45, 20.00, 85.83, 7.11, 112.33
predicted_crop = test_recommendation(N, P, K, temperature, humidity, ph, rainfall)
print(f"Predicted Crop ID: {predicted_crop}")
