from collections import defaultdict
class SessionMemory:
    def __init__(self): self.sessions = defaultdict(list)
    def add(self, session_id, role, text): self.sessions[session_id].append({"role": role, "text": text})
    def history(self, session_id): return list(self.sessions[session_id])
    def reset(self, session_id): self.sessions.pop(session_id, None)
