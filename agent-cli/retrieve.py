from .store import connect, top_k
from .providers import OpenAIProvider

SYSTEM_PROMPT = """You are a precise repository assistant.
- Use only provided context to answer.
- Always cite files with path:line ranges like path:start-end.
- If unsure, say you need more context and suggest where to look.
"""

def answer(db_path: str, provider: OpenAIProvider, question: str, k=12):
    con = connect(db_path)
    qvec = provider.embed([question])[0]
    hits = top_k(con, qvec, k=k)
    context = []
    for _, (path, s, e, content), score in hits:
        context.append(f"[{path}:{s}-{e}]\n{content}")
    user = f"Question: {question}\n\nContext:\n" + "\n\n---\n\n".join(context[:8])
    return provider.chat(SYSTEM_PROMPT, user), hits
