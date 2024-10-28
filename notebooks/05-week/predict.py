import pickle
from flask import Flask, jsonify
from flask import request

model_path = 'model_C=1.0.bin'

with open(model_path, 'rb') as file:
    (dv, model) = pickle.load(file)

app = Flask('churn')


@app.route('/predict', methods=['POST'])
def predict():

    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred <=0.5
    result = {
            'churn_probability' : y_pred,
            'churn' : bool(churn)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)