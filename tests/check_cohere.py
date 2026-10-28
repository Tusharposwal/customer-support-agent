import os

import cohere
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("COHERE_API_KEY")

if not api_key:
    raise RuntimeError("COHERE_API_KEY is missing. Check your project-root .env file.")

client = cohere.ClientV2(api_key=api_key)

response = client.embed(
    texts=["My package has not arrived"],
    model="embed-v4.0",
    input_type="search_query",
    embedding_types=["float"],
)

embedding = response.embeddings.float[0]

print("Cohere API connection successful")
print("Embedding dimensions:", len(embedding))
print("First 5 values:", embedding[:5])
