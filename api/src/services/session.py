import uuid
from datetime import datetime
from src.services.database import DatabaseService

class SessionService:
    def __init__(self):
        self.db = DatabaseService()

    def create_session(self, file_id: str = None) -> str:
        """Create a new chat session"""
        session_id = str(uuid.uuid4())
        current_time = datetime.utcnow()

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO sessions (session_id, created_at, last_updated, file_id)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, current_time, current_time, file_id)
            )
            conn.commit()
        return session_id

    def get_session(self, session_id: str) -> dict:
        """Get session details"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            session = cursor.fetchone()
        return session

    def get_file_id(self, session_id: str) -> str:
        """Get file id from session id"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT file_id FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            file_id = cursor.fetchone()
        return file_id

