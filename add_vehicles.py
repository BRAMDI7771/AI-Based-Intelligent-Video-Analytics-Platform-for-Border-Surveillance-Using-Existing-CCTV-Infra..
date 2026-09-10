import sqlite3
from pathlib import Path

# ============================================================
# DATABASE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "surveillance.db"


# ============================================================
# CONNECT DATABASE
# ============================================================

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()


# ============================================================
# VEHICLE DATA
# ============================================================

vehicles = [
    ("V001", "UP30N3639", "Car", "Alto", "Silver", "P001"),
    ("V002", "UP35BW7275", "Car", "Brezza", "White", "P002"),
    ("V003", "UP27BC0505", "Car", "Scorpio", "Black", "P001"),
    ("V004", "UP30U2714", "Bike", "Splendor", "Black", "P003"),
    ("V005", "UP70BP7307", "Car", "Hyundai", "Silver", "P004"),
    ("V006", "UP30BD6536", "Scooter", "Maestro", "Black", "P005"),
    ("V007", "UP32QH4705", "Car", "Verna", "Black", "P006"),
    ("V008", "UP32PE1275", "Car", "Brezza", "White", "P002"),
]


# ============================================================
# INSERT VEHICLES
# ============================================================

for vehicle_code, registration_number, vehicle_type, model, color, owner in vehicles:

    cursor.execute("""
        INSERT OR IGNORE INTO vehicles
        (
            vehicle_code,
            registration_number,
            vehicle_type,
            model,
            color,
            owner_person_code,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        vehicle_code,
        registration_number,
        vehicle_type,
        model,
        color,
        owner,
        "AUTHORIZED"
    ))


# ============================================================
# SAVE
# ============================================================

conn.commit()


# ============================================================
# DISPLAY DATA
# ============================================================

print("\n======================================")
print(" 8 VEHICLES ADDED SUCCESSFULLY")
print("======================================")

cursor.execute("""
    SELECT
        vehicle_code,
        registration_number,
        vehicle_type,
        model,
        color,
        owner_person_code,
        status
    FROM vehicles
    ORDER BY id
""")

for row in cursor.fetchall():
    print(row)


# ============================================================
# CLOSE
# ============================================================

conn.close()

print("======================================")