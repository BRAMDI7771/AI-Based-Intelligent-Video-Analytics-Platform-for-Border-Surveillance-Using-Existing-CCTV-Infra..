import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "surveillance.db"

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Persons
cursor.execute("""
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_code TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    role TEXT,
    photo_path TEXT,
    status TEXT DEFAULT 'AUTHORIZED',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

# Vehicles
cursor.execute("""
CREATE TABLE IF NOT EXISTS vehicles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_code TEXT UNIQUE NOT NULL,
    registration_number TEXT UNIQUE,
    vehicle_type TEXT,
    model TEXT,
    color TEXT,
    owner_person_code TEXT,
    status TEXT DEFAULT 'AUTHORIZED',
    photo_path TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (owner_person_code)
    REFERENCES persons(person_code)
)
""")

# Events
cursor.execute("""
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    person_code TEXT,
    vehicle_code TEXT,
    camera_id TEXT,
    track_id INTEGER,
    description TEXT,
    screenshot_path TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

print("======================================")
print(" DATABASE CREATED SUCCESSFULLY")
print("======================================")
print(f"Database: {DB_PATH}")
print("Tables:")
print("  1. persons")
print("  2. vehicles")
print("  3. events")
print("======================================")