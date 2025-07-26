from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd

# Load trained model and scalers
model = pickle.load(open('model.pkl', 'rb'))  # Random Forest Model
minmax_scaler = pickle.load(open('minmaxscaler.pkl', 'rb'))  # MinMaxScaler
standard_scaler = pickle.load(open('standscaler.pkl', 'rb'))  # StandardScaler

# Flask App
app = Flask(__name__, template_folder='templates')

# Crop Mapping Dictionary with Descriptions and Pesticides
crop_mapping = {
    1: {"name": "Rice",
        "description": "Rice is a staple food crop grown in flooded fields, requiring high humidity and temperature.",
        "pesticides": ["Carbofuran (seed treatment)", "Triazophos (vegetative stage)", "Chlorpyrifos (grain filling)"]},

    2: {"name": "Maize",
        "description": "Maize is a cereal grain that thrives in well-drained soil and requires moderate rainfall.",
        "pesticides": ["Atrazine (pre-emergence)", "Lambda-cyhalothrin (mid-growth)", "Chlorantraniliprole (flowering stage)"]},

    3: {"name": "Jute",
        "description": "Jute is a fiber crop cultivated in warm and humid climates, primarily used for making bags and ropes.",
        "pesticides": ["Imidacloprid (early stage)", "Mancozeb (growth stage)", "Lambda-cyhalothrin (pre-harvest)"]},

    4: {"name": "Cotton",
        "description": "Cotton is a soft fiber crop that needs long frost-free periods and moderate rainfall.",
        "pesticides": ["Imidacloprid (seed treatment)", "Acephate (bollworm control)", "Chlorpyrifos (flowering stage)"]},

    5: {"name": "Coconut",
        "description": "Coconut is a tropical fruit grown in sandy coastal regions with high humidity.",
        "pesticides": ["Neem Oil (organic)", "Chlorpyrifos (weevil control)", "Copper oxychloride (fungal diseases)"]},

    6: {"name": "Papaya",
        "description": "Papaya is a tropical fruit that requires warm temperatures and well-drained soil.",
        "pesticides": ["Imidacloprid (aphid control)", "Sulfur (powdery mildew)", "Copper fungicide (anthracnose control)"]},

    7: {"name": "Orange",
        "description": "Oranges are citrus fruits that require well-drained soil and warm temperatures.",
        "pesticides": ["Copper oxychloride (fungal control)", "Lambda-cyhalothrin (insect control)", "Neem oil (organic option)"]},

    8: {"name": "Apple",
        "description": "Apples require a cool climate and well-drained soil with sufficient chilling hours for fruit set.",
        "pesticides": ["Mancozeb (scab control)", "Captan (fungal infections)", "Chlorpyrifos (codling moth)"]},

    9: {"name": "Muskmelon",
        "description": "Muskmelon is a warm-season fruit that grows best in sandy loam soil with good drainage.",
        "pesticides": ["Copper fungicide (downy mildew)", "Imidacloprid (whitefly control)", "Sulfur (powdery mildew)"]},

    10: {"name": "Watermelon",
        "description": "Watermelon is a summer fruit that grows best in sandy, well-drained soil with ample sunlight.",
        "pesticides": ["Mancozeb (anthracnose control)", "Chlorpyrifos (insect control)", "Sulfur (powdery mildew)"]},

    11: {"name": "Grapes",
        "description": "Grapes are cultivated in temperate climates and require proper trellising and pruning.",
        "pesticides": ["Sulfur (powdery mildew)", "Mancozeb (black rot control)", "Imidacloprid (leafhopper control)"]},

    12: {"name": "Mango",
        "description": "Mango trees grow best in tropical climates with dry winters and deep, well-drained soil.",
        "pesticides": ["Chlorpyrifos (stem borer control)", "Carbendazim (anthracnose control)", "Neem oil (organic option)"]},

    13: {"name": "Banana",
        "description": "Banana is a tropical fruit that needs warm temperatures, high humidity, and loamy soil.",
        "pesticides": ["Carbendazim (fungal control)", "Chlorpyrifos (borer control)", "Sulfur (leaf spot control)"]},

    14: {"name": "Pomegranate",
        "description": "Pomegranate is a drought-tolerant fruit crop that thrives in arid and semi-arid regions.",
        "pesticides": ["Copper oxychloride (fungal control)", "Imidacloprid (aphid control)", "Sulfur (powdery mildew)"]},

    15: {"name": "Lentil",
        "description": "Lentils are legume crops grown in cool-season climates and require well-drained soil.",
        "pesticides": ["Metalaxyl (root rot control)", "Imidacloprid (aphid control)", "Mancozeb (leaf spot)"]},

    16: {"name": "Blackgram",
        "description": "Blackgram is a legume that grows in warm, semi-arid climates with moderate rainfall.",
        "pesticides": ["Carbendazim (powdery mildew)", "Imidacloprid (whitefly control)", "Thiamethoxam (aphid control)"]},

    17: {"name": "Mungbean",
        "description": "Mungbean is a short-duration legume crop that requires sandy loam soil and warm weather.",
        "pesticides": ["Mancozeb (fungal diseases)", "Neem oil (organic control)", "Carbofuran (nematode control)"]},

    18: {"name": "Mothbeans",
        "description": "Mothbean is a drought-resistant legume crop grown in dry, arid regions.",
        "pesticides": ["Chlorpyrifos (pod borer control)", "Mancozeb (root rot control)", "Sulfur (powdery mildew)"]},

    19: {"name": "Pigeonpeas",
        "description": "Pigeonpea is a hardy legume crop that can withstand drought conditions.",
        "pesticides": ["Carbendazim (wilt control)", "Neem oil (organic control)", "Thiamethoxam (leafhopper control)"]},

    20: {"name": "Kidneybeans",
        "description": "Kidney beans require well-drained loamy soil with good moisture retention.",
        "pesticides": ["Mancozeb (rust control)", "Imidacloprid (beetle control)", "Chlorpyrifos (aphid control)"]},

    21: {"name": "Chickpea",
        "description": "Chickpea is a cool-season crop that grows in semi-arid regions with well-drained soil.",
        "pesticides": ["Carbendazim (fungal wilt control)", "Imidacloprid (whitefly control)", "Thiamethoxam (beetle control)"]},

    22: {"name": "Coffee",
        "description": "Coffee is a tropical crop grown in high-altitude regions with well-drained soil.",
        "pesticides": ["Copper oxychloride (fungal control)", "Neem oil (organic option)", "Chlorpyrifos (insect control)"]}
}

def recommend_crop(features):
    """
    Function to predict the best crop based on input features.
    """
    try:
        # Define feature names exactly as used during training
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']  # Match names precisely

        # Convert features to DataFrame with correct column names
        features_df = pd.DataFrame([features], columns=feature_names)

        # Apply MinMax Scaling and Standard Scaling
        scaled_features = minmax_scaler.transform(features_df)  # Apply MinMax Scaling
        final_features = standard_scaler.transform(scaled_features)  # Apply Standard Scaling

        # Predict crop using Random Forest
        prediction = model.predict(final_features)
        crop_id = int(prediction[0])

        # Retrieve crop name from dictionary
        crop_data = crop_mapping.get(crop_id, {"name": "Unknown", "description": "No data available.", "pesticides": []})
        return crop_data
    except Exception as e:
        return {"name": "Error", "description": str(e), "pesticides": []}


@app.route('/')
def index():
    return render_template("index.html", result=None, description=None, pesticides=None)

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Get input values from form
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])  # Fixed spelling mistake
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        # Store input as feature list
        feature_list = [N, P, K, temp, humidity, ph, rainfall]

        # Predict crop
        crop_data = recommend_crop(feature_list)

        # Display result
        result = f"Recommended Crop: {crop_data['name']} 🌱 is the best crop to be cultivated."
        description = crop_data['description']
        pesticides = crop_data['pesticides']
    except Exception as e:
        result = f"Error: {str(e)}"
        description = None
        pesticides = None

    return render_template('index.html', result=result, description=description, pesticides=pesticides)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
