import requests
import urllib.request

def url_ok(url, timeout):
    try:
        r = requests.head(url, timeout=timeout)
        return r.status_code == 200
    except:
        return False