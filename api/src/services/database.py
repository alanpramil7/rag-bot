import sqlite3
from src.utils.logger import logger
from src.config import settings

class DatabaseService:
    """Service class for database operations"""

    def __init__(self):
        """
        Initialize database service

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = settings.DB_NAME
        self._initialize_db()

    def _initialize_db(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at TIMESTAMP,
                        last_updated TIMESTAMP,
                        file_id TEXT NULL
                    )
                ''')

                # Create messages table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS messages (
                        message_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        role TEXT,
                        content TEXT,
                        timestamp TIMESTAMP,
                        metadata TEXT,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    )
                ''')

                conn.commit()
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
