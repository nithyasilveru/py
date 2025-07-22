from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model and column data
model = pickle.load(open("model/banglore_home_prices_model.pickle", "rb"))
columns = pickle.load(open("model/columns.pickle", "rb"))  # or JSON if you're using .json
locations = columns[3:]  # first 3 are sqft, bath, bhk

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sqft = float(data['sqft'])
    bath = int(data['bath'])
    bhk = int(data['bhk'])
    location = data['location']

    x = np.zeros(len(columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if location in locations:
        loc_index = columns.index(location)
        x[loc_index] = 1

    prediction = model.predict([x])[0]
    return jsonify({'estimated_price_lakhs': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)
