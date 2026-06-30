import requests
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://openreview.net/forum?id=0LZRtvK871",
    "Accept": "application/pdf,*/*",
}
resp = requests.get("https://openreview.net/pdf?id=0LZRtvK871", headers=headers, timeout=20)
print(resp.status_code, resp.headers.get("content-type"), len(resp.content))
print(resp.content[:20])