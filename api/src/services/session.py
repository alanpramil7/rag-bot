import uuid
from datetime import datetime
from typing import List
from src.utils.logger import logger
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

    def insert_file_id(self, session_id: str, file_id: str):
        """ Insert new file_id for exisiting session """
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

    def get_file_id(self, session_id: str) -> List[str]:
        """Get file id from session id"""
        try:
            logger.debug("Getting file-ids")
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT file_id FROM sessions WHERE session_id = ?",
                    (session_id,)
                )
                file_ids = [row[0] for row in cursor.fetchall()]
            return file_ids
        except Exception as e:
            logger.debug(f"Error getting file id: {str(e)}")
