import sqlite3
from typing import Optional
from pathlib import Path


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def sqlite_query(db_path: str, query: str) -> str:
    """
    Execute a SELECT query on a SQLite database and return the results.

    Example:
    sqlite_query("test.db", "SELECT * FROM users")
    """
    try:
        conn = get_connection(Path(db_path))
        cursor = conn.execute(query)

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "No results."

        return "\n".join([str(dict(row)) for row in rows])

    except Exception as e:
        return f"Error: {str(e)}"


def sqlite_execute(db_path: str, query: str) -> str:
    """
    Execute a SQL command (CREATE, INSERT, UPDATE, DELETE) on a SQLite database.

    Example:
    sqlite_execute("test.db", "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    """
    try:
        conn = get_connection(Path(db_path))
        conn.execute(query)
        conn.commit()
        conn.close()

        return "Query executed successfully."

    except Exception as e:
        return f"Error: {str(e)}"
