import uuid
from datetime import datetime
from src.services.database import DatabaseService

class ChatService:
    def __init__(self):
        self.db = DatabaseService()

    def save_message(self, session_id: str, role: str, content: str, metadata: str = None):
        """Save a message to the chat history"""
        message_id = str(uuid.uuid4())
        current_time = datetime.utcnow()

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO messages (message_id, session_id, role, content, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (message_id, session_id, role, content, current_time, metadata)
            )
            cursor.execute(
                """
                UPDATE sessions SET last_updated = ? WHERE session_id = ?
                """,
                (current_time, session_id)
            )
            conn.commit()

    def get_chat_history(self, session_id: str) -> list:
        """Get chat history for a session"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT role, content FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
                """,
                (session_id,)
            )
            messages = cursor.fetchall()
        return [{"role": msg[0], "content": msg[1]} for msg in messages]
