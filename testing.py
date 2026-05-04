import os
path = "data/abbreviations.json"
print("Looking for:", os.path.abspath(path))
print("Exists:", os.path.exists(path))
if os.path.exists(path):
    with open(path, "rb") as f:
        raw = f.read()[:100]
        print("First 100 bytes (repr):", repr(raw))