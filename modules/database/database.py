"""
Database connection and initialization module.
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import time
from typing import Optional, Dict, Any
from contextlib import contextmanager

from ..config import logger

# PostgreSQL connection parameters from environment variables
PG_HOST = os.environ.get('MBZUAI_PG_HOST', 'localhost')
PG_PORT = os.environ.get('MBZUAI_PG_PORT', '5432')
PG_USER = os.environ.get('MBZUAI_PG_USER', 'postgres')
PG_PASSWORD = os.environ.get('MBZUAI_PG_PASSWORD', 'postgres')
PG_DATABASE = os.environ.get('MBZUAI_PG_DATABASE', 'mbzuai_database')  # Using a dedicated database

# Since we're using a dedicated database, we don't need table prefixes anymore
# but we'll keep the variable for backward compatibility
TABLE_PREFIX = ''

# Log database connection info (without password)
logger.info(f"Connecting to PostgreSQL database: {PG_DATABASE} on {PG_HOST}:{PG_PORT}")

@contextmanager
def get_db_connection():
    """
    Get a connection to the PostgreSQL database.
    Uses a context manager pattern for automatic connection cleanup.
    
    Yields:
        Connection: PostgreSQL database connection
    """
    conn = None
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASSWORD,
            database=PG_DATABASE
        )
        # Use RealDictCursor to return rows as dictionaries (similar to SQLite's Row factory)
        conn.cursor_factory = RealDictCursor
        # Yield the connection to the caller
        yield conn
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if conn is not None:
            conn.close()
            logger.debug("Database connection closed")

def backup_database() -> None:
    """
    Create a backup of the PostgreSQL database.
    
    Note: This is a placeholder. For PostgreSQL, backups are typically handled through:
    1. pg_dump for logical backups
    2. Automated backup solutions provided by the hosting service
    3. Replication and point-in-time recovery for enterprise setups
    
    This function logs a reminder that backups should be configured at the database level.
    """
    logger.info("PostgreSQL backups should be configured at the database server level.")
    logger.info("Consider using pg_dump, automated backup solutions, or consulting your DBA.")


def init_db() -> None:
    """
    Initialize the PostgreSQL database by creating tables if they don't exist.
    Creates required indices for better performance.
    """
    # Log a reminder about backups
    backup_database()
    
    # Initialize database tables and indices
    with get_db_connection() as conn:
        try:
            cursor = conn.cursor()
            
            # Create chat_history table with PostgreSQL-specific types
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                feedback TEXT CHECK(feedback IN ('like', 'dislike', NULL)),
                id_str TEXT UNIQUE  -- Added field for custom string IDs
            )
            ''')
            
            # Create sources table with foreign key relationship to chat_history
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS sources (
                id SERIAL PRIMARY KEY,
                chat_id INTEGER NOT NULL,
                url TEXT NOT NULL,
                cite_num TEXT NOT NULL,
                FOREIGN KEY (chat_id) REFERENCES chat_history (id) ON DELETE CASCADE
            )
            ''')
            
            # Create indices for better query performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_history_timestamp ON chat_history(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sources_chat_id ON sources(chat_id)')
            
            # Check schema version and perform migrations if needed
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Check current schema version
            cursor.execute('SELECT MAX(version) FROM schema_version')
            result = cursor.fetchone()
            current_version = result['max'] if result and result['max'] is not None else 0
            
            # Migration: Add id_str column if it doesn't exist (schema version 2)
            if current_version < 2:
                try:
                    # Check if the id_str column already exists
                    cursor.execute("""SELECT column_name FROM information_schema.columns 
                                    WHERE table_name='chat_history' AND column_name='id_str'""")
                    if not cursor.fetchone():
                        # Add the id_str column if it doesn't exist
                        cursor.execute('ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS id_str TEXT UNIQUE')
                        logger.info("Added id_str column to chat_history table")
                    
                    # Update schema version
                    cursor.execute('INSERT INTO schema_version (version) VALUES (%s)', (2,))
                    logger.info("Applied schema migration to version 2")
                except Exception as e:
                    logger.error(f"Error during migration to version 2: {e}")
                    raise
            
            # Update schema version if migrations were applied
            if current_version < 1:  # Current schema version
                cursor.execute('INSERT INTO schema_version (version) VALUES (%s)', (1,))
            
            conn.commit()
            logger.info("Database initialized successfully")
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Database initialization error: {e}")
            # Don't raise here - we want the app to continue even if DB init fails
        except Exception as e:
            conn.rollback()
            logger.error(f"Unexpected error during database initialization: {e}")
