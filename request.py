import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={	'age':5, 'menopause':1, 'tumor-size':1, 'inv-nodes':1, 'node-caps':2,
       'deg-malig':1, 'breast':3, 'breast-quad':1, 'irradiat':1})

print(r.json())