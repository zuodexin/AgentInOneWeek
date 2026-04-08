from tools.database import sqlite_execute, sqlite_query
from tools.asr import transcribe


all_tools = [sqlite_query, sqlite_execute, transcribe]
