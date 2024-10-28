# from flask import Flask
#
# app = Flask('ping')
#
# @app.route('/ping',methods = ['GET'])
# def ping():
#     return ("Pong")
#
#
#
# if __name__=="__main__":
#     app.run(debug=True,host='0.0.0.0',port = 9696)

import pickle

model_path = 'model1.bin'
dv_path = 'dv.bin'

with open(model_path, 'rb') as file:
    model = pickle.load(file)
with open(dv_path, 'rb') as file:
    dv = pickle.load(file)


def predict(customer):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    return (y_pred)

if __name__ == "__main__":
    responses = predict({"job": "management", "duration": 400, "poutcome": "success"})
    print(responses)
