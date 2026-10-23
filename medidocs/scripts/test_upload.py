import urllib.request
import json
import uuid
import sys

if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')

boundary = '----WebKitFormBoundary' + uuid.uuid4().hex
crlf = '\r\n'

parts = [
    f'--{boundary}',
    'Content-Disposition: form-data; name="file"; filename="test_policy.txt"',
    'Content-Type: text/plain',
    '',
    'Section 1: Infection Control Guideline.\nAll personnel must sanitize hands before entering ICU.\n\nSection 2: Protective Equipment.\nN95 respirators required in isolation rooms.',
    f'--{boundary}',
    'Content-Disposition: form-data; name="title"',
    '',
    'ICU Infection Control Protocols',
    f'--{boundary}',
    'Content-Disposition: form-data; name="code"',
    '',
    'POL-ICU-999',
    f'--{boundary}',
    'Content-Disposition: form-data; name="department"',
    '',
    'Intensive Care',
    f'--{boundary}',
    'Content-Disposition: form-data; name="access_level"',
    '',
    'CLINICAL',
    f'--{boundary}--',
    ''
]

body = crlf.join(parts).encode('utf-8')
req = urllib.request.Request('http://127.0.0.1:8000/api/documents/upload', data=body, method='POST')
req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')

resp = urllib.request.urlopen(req)
data = json.loads(resp.read().decode('utf-8'))
print('[✓] Upload status:', data['status'])
print('[✓] Chunks indexed:', data['chunks_indexed'])
print('[✓] Total documents now:', data['total_documents'])
print('[✓] Document title:', data['document']['title'])
