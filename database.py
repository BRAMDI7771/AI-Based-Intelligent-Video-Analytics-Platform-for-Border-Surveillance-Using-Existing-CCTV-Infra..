import json
import os

DB_FILE = "data/persons_db.json"
os.makedirs("data", exist_ok=True)

if not os.path.exists(DB_FILE):
    with open(DB_FILE, "w") as f:
        json.dump({}, f)

def get_person_info(person_id):
    with open(DB_FILE, "r") as f:
        db = json.load(f)
    return db.get(str(person_id), {"name": "Unknown", "status": "Unidentified"})

def add_person_info(person_id, name, status="Verified"):
    with open(DB_FILE, "r") as f:
        db = json.load(f)
    db[str(person_id)] = {"name": name, "status": status}
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=2)
