import json, glob

hadith_files = glob.glob("hadith-json/bukhari/*.json")

with open("hadith.txt", "w") as out:
    for file in hadith_files:
        with open(file) as f:
            data = json.load(f)
            for h in data.get("hadiths", []):
                text = f"Sahih Bukhari {h['hadithnumber']} — {h['english']}\n"
                out.write(text)
