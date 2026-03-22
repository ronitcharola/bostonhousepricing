import pickle
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np

app = Flask(__name__)

# Load the model
regmodel = pickle.load(open('regmodel.pkl','rb'))
scaler = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    return jsonify(output[0])


# ✅ FIXED: properly defined route
@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text=f"The House prediction price is {output}")


if __name__ == "__main__":
    app.run(debug=True)