-- Media sources (video/audio files)
CREATE TABLE IF NOT EXISTS media_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL UNIQUE,
    media_type TEXT NOT NULL CHECK(media_type IN ('video', 'audio', 'song')),
    duration_sec REAL,
    sample_rate INTEGER,
    channels INTEGER,
    metadata_json TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Extracted tracks from media
CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES media_sources(id),
    track_type TEXT NOT NULL CHECK(track_type IN ('video', 'audio', 'vocal', 'accompaniment', 'drums', 'bass', 'other')),
    file_path TEXT NOT NULL,
    start_sec REAL NOT NULL DEFAULT 0,
    end_sec REAL,
    duration_sec REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Segments (speech/singing segments within a track)
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL REFERENCES tracks(id),
    source_id INTEGER NOT NULL REFERENCES media_sources(id),
    speaker_id TEXT,
    start_sec REAL NOT NULL,
    end_sec REAL NOT NULL,
    text TEXT,
    confidence REAL,
    segment_type TEXT DEFAULT 'speech' CHECK(segment_type IN ('speech', 'singing', 'silence', 'noise')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Word-level alignments
CREATE TABLE IF NOT EXISTS word_alignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id INTEGER NOT NULL REFERENCES segments(id),
    word TEXT NOT NULL,
    start_sec REAL NOT NULL,
    end_sec REAL NOT NULL,
    confidence REAL,
    pitch_hz REAL,
    note TEXT
);

-- Speaker registry
CREATE TABLE IF NOT EXISTS speakers (
    id TEXT PRIMARY KEY,
    name TEXT,
    source_id INTEGER REFERENCES media_sources(id),
    embedding_path TEXT,
    metadata_json TEXT DEFAULT '{}'
);