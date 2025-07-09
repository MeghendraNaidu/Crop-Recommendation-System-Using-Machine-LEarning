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
        self.assertEqual(recommend(88, 38, 15, 25.08, 65.92, 6.45, 62.49), 2)

    def test_maize(self):
        """Test case expected to recommend Maize"""
        self.assertEqual(recommend(25, 76, 24 , 15.33, 24.91, 5.56, 135.33), 20)

    def test_coffee(self):
        """Test case expected to recommend Coffee"""
        self.assertEqual(recommend(39, 37, 15, 28.99, 83.78, 6.82, 59.84), 17)

if __name__ == '__main__':
    unittest.main()
