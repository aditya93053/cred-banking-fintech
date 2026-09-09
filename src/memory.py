from collections import defaultdict
from langchain_core.chat_history import InMemoryChatMessageHistory


class SessionMemory:
    def __init__(self):
        self.sessions = defaultdict(
            InMemoryChatMessageHistory
        )

    def add(self, session_id, role, text):
        history = self.sessions[session_id]

        if role == "user":
            history.add_user_message(text)
        else:
            history.add_ai_message(text)

    def history(self, session_id):
        history = self.sessions[session_id]

        return [
            {
                "role": message.type,
                "text": message.content,
            }
            for message in history.messages
        ]

    def reset(self, session_id):
        self.sessions.pop(
            session_id,
            None,
        )

    def clear(self):
        self.sessions.clear()

    def session_exists(self, session_id):
        return session_id in self.sessions
