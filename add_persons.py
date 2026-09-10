import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "surveillance.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

persons = [
    ("P001", "Person 1", "Member", "database/persons/person_1/akshat"),
    ("P002", "Person 2", "Member", "database/persons/person_2/anuj"),
    ("P003", "Person 3", "Member", "database/persons/person_3/abhay"),
    ("P004", "Person 4", "Member", "database/persons/person_4/harshit"),
    ("P005", "Person 5", "Member", "database/persons/person_5/aditi"),
    ("P006", "Person 6", "Member", "database/persons/person_6/parul"),
]

for person_code, name, role, photo_path in persons:

    cursor.execute("""
        INSERT OR IGNORE INTO persons
        (person_code, name, role, photo_path, status)
        VALUES (?, ?, ?, ?, ?)
    """, (
        person_code,
        name,
        role,
        photo_path,
        "AUTHORIZED"
    ))

conn.commit()

print("\n======================================")
print(" 6 PERSONS ADDED SUCCESSFULLY")
print("======================================")

cursor.execute("""
    SELECT person_code, name, role, photo_path, status
    FROM persons
""")

for row in cursor.fetchall():
    print(row)

conn.close()