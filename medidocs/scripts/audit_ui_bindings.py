import re

with open(r"d:\genai\GenAI\medidocs\frontend\index.html", encoding="utf-8") as f:
    html = f.read()

with open(r"d:\genai\GenAI\medidocs\frontend\app.js", encoding="utf-8") as f:
    js = f.read()

html_ids = set(re.findall(r'id=["\']([^"\']+)["\']', html))
js_ids = set(re.findall(r'getElementById\(["\']([^"\']+)["\']\)', js))

missing_in_html = [i for i in js_ids if i not in html_ids]
print(f"Total HTML IDs: {len(html_ids)}")
print(f"Total getElementById targets in JS: {len(js_ids)}")
print(f"IDs queried by getElementById that DO NOT exist in HTML ({len(missing_in_html)}):")
for m in sorted(missing_in_html):
    print(f"  - {m}")
