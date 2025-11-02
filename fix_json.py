with open('travel_destination.ipynb', 'rb') as f:
    content = f.read()
content = content.decode('utf-8', errors='ignore')
if content.startswith('tha{'):
    content = content.replace('tha{', '{', 1)
with open('travel_destination.ipynb', 'w', encoding='utf-8') as f:
    f.write(content)
