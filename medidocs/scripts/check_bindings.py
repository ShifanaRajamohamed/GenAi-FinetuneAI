import re

with open(r"d:\genai\GenAI\medidocs\frontend\index.html", encoding="utf-8") as f:
    html = f.read()
ids = re.findall(r'id=["\']([^"\']+)["\']', html)
print(f"Total IDs found in index.html: {len(ids)}")

with open(r"d:\genai\GenAI\medidocs\frontend\app.js", encoding="utf-8") as f:
    js = f.read()

unhandled = []
for i in ids:
    if i not in js:
        unhandled.append(i)

print(f"IDs not referenced in app.js ({len(unhandled)}):")
for u in unhandled:
    print(f"  - {u}")
