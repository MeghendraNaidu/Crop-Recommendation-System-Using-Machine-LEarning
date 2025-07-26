import unittest
import pickle
import numpy as np
import pandas as pd

# Load model and scalers
randclf = pickle.load(open('model.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))

# Define feature names (same as training data)
feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

def recommend(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Convert to DataFrame with feature names
    features_df = pd.DataFrame(features, columns=feature_names)

    # Apply scaling with proper feature names
    features_scaled = mx.transform(features_df)
    features_final = sc.transform(features_scaled)

    return randclf.predict(features_final)[0]

class TestCropRecommendation(unittest.TestCase):
    def test_rice(self):
        """Test case expected to recommend Rice"""
        self.assertEqual(recommend(33, 23, 45, 20.00, 85.83, 7.11, 112.33), 14)

    def test_maize(self):
        """Test case expected to recommend Maize"""
        self.assertEqual(recommend(15, 77, 20, 25.13, 66.92, 7.39, 49.04), 15)

    def test_coffee(self):
        """Test case expected to recommend Coffee"""
        self.assertEqual(recommend(44, 75, 22, 30.03, 64.14, 7.57, 71.21), 16)

if __name__ == '__main__':
    unittest.main()
