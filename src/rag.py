from .knowledge_base import KB
from .guardrails import in_scope

class LocalRAG:
    def __init__(self, strategy="sentence"):
        self.strategy = strategy
        self.docs = KB

    def retrieve(self, query, k=3):
        q = set(query.lower().split())
        scored = []
        for key, text in self.docs.items():
            overlap = len(q & set(text.lower().replace(",","").split()))
            scored.append((overlap, key, text))
        scored.sort(reverse=True)
        return [{"topic": k, "text": t, "score": float(s)} for s,k,t in scored[:k]]

    def answer(self, query):
        if not in_scope(query):
            return None, []
        hits = self.retrieve(query)
        if not hits or hits[0]["score"] <= 0:
            return None, []
        return "Based only on the local Cred support knowledge base: " + hits[0]["text"], [h["topic"] for h in hits]
