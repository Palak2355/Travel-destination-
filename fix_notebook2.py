with open('travel_destination.ipynb', 'rb') as f:
    content = f.read()
content = content.decode('utf-8', errors='ignore')
print(content[:200])
