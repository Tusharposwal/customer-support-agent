import json
from pathlib import Path

kb_path = Path("data/kb.json")

with kb_path.open("r", encoding="utf-8") as file:
    articles = json.load(file)

print(f"Loaded {len(articles)} articles")

for article in articles:
    print(article["id"], "-", article["title"])