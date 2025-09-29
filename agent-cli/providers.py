import os, numpy as np
from openai import OpenAI

class OpenAIProvider:
    def __init__(self, chat_model="gpt-4o-mini", embedding_model="text-embedding-3-small"):
        self.client = OpenAI()
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        res = self.client.embeddings.create(model=self.embedding_model, input=texts)
        return [d.embedding for d in res.data]

    def chat(self, system: str, user: str):
        res = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.2
        )
        return res.choices[0].message.content
