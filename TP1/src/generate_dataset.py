
import urllib.request
import os

img_dir = "TP1/data/images"
to_download = 12

for i in range(to_download):
    filename = f"{img_dir}/random_{i}.jpg"
    url = f"https://picsum.photos/800/600?random={i}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            with open(filename, 'wb') as f:
                f.write(response.read())
        print(f"✅ Image aléatoire {i} ajoutée")
    except:
        print(f"❌ Erreur sur l'image {i}")