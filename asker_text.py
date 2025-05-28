import json
import requests as rq

url = 'http://localhost:8087/ask'
reqHeader = {'Content-Type': 'application/json'}
while True:
    question = input('Next question:')
    resp = rq.post(url, headers = reqHeader, json={"question": question}, verify=False)
    print('Done:', resp.status_code)
