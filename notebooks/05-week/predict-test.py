import requests
url = 'http://localhost:9696/predict'

customer = {
  "gender": "female",
  "seniorcitizen": 0,
  "partner": "yes",
  "dependents": "no",
  "tenure": 1,
  "phoneservice": "no",
  "multiplelines": "no_phone_service",
  "internetservice": "dsl",
  "onlinesecurity": "no",
  "onlinebackup": "yes",
  "deviceprotection": "no",
  "techsupport": "no",
  "streamingtv": "no",
  "streamingmovies": "no",
  "contract": "month-to-month",
  "paperlessbilling": "yes",
  "paymentmethod": "electronic_check",
  "monthlycharges": 29.85,
  "totalcharges": 29.85
}
responses = requests.post(url,json=customer).json()
print(responses)

customer = "7590-vhveg"
if responses['churn']:
    print(f'send an email to the customer ID {customer}')
else:
    print(f'not sending an email to the customer ID{customer}')