import os
import warnings

warnings.filterwarnings('ignore')

# Ignore broken shell-level proxy settings for this one-off script.
for proxy_var in (
    'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
    'http_proxy', 'https_proxy', 'all_proxy',
    'GIT_HTTP_PROXY', 'GIT_HTTPS_PROXY'
):
    os.environ.pop(proxy_var, None)

import requests
import urllib3
from urllib3.util.ssl_ import create_urllib3_context

urllib3.disable_warnings()
requests.Session.trust_env = False

# Force requests-based clients to skip TLS verification.
_original_request = requests.Session.request

def patched_request(self, *args, **kwargs):
    kwargs.setdefault('verify', False)
    return _original_request(self, *args, **kwargs)

requests.Session.request = patched_request

# Keep urllib3 contexts permissive too, for libraries that build them directly.
def patched_create_urllib3_context(*args, **kwargs):
    context = create_urllib3_context(*args, **kwargs)
    context.check_hostname = False
    context.verify_mode = 0  # ssl.CERT_NONE
    return context

urllib3.util.ssl_.create_urllib3_context = patched_create_urllib3_context

import openreview

client = openreview.api.OpenReviewClient(
    baseurl='https://api2.openreview.net',
    username='harshadk@aero.iitb.ac.in',
    password='Redacted'
)

note = client.get_note(id='MiV3WXDYJb')
authorids = note.content['authorids']['value'][:2]
profiles = openreview.tools.get_profiles(client, authorids)

for profile in profiles:
    history = profile.content.get('history', [])
    current = next((h for h in history if h.get('end') is None), history[0] if history else None)
    if current:
        inst = current.get('institution', {})
        print(f"{profile.id}: {inst.get('name')}, {inst.get('country')}")

        