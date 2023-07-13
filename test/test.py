import requests

url = 'http://localhost:8000/predict'
files = {'image': open('mnist.png', 'rb')}
response = requests.post(url, files=files)
print(response.content)
