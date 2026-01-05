import requests
from settings import API_MARKER_ENDPOINT

res = requests.post(API_MARKER_ENDPOINT, data={"output_format":"markdown", "file_url": "https://arxiv.org/pdf/2006.11239"})
print(res.json())