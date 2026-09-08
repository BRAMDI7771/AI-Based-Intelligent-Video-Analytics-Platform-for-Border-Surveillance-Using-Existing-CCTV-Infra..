import json
import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "app.db")


def clear_and_add_people(people_list):
  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()

  # Naye 6 members add karne se pehle purana clear kar rahe hain
  cursor.execute("DELETE FROM persons")

  for person in people_list:
    name = person["name"]
    img_path = person["image_path"]
    embedding_json = json.dumps(person["embedding"])

    cursor.execute(
        """
            INSERT INTO persons (name, image_path, embedding)
            VALUES (?, ?, ?)
        """,
        (name, img_path, embedding_json),
    )

    print(f"Added: {name} -> {img_path}")

  conn.commit()
  conn.close()
  print("\nTotal 6 members database me successfully save ho gaye!")


if __name__ == "__main__":
  # Apne 6 members ke actual names aur image paths yahan edit kar sakte ho:
  people_data = [
      {
          "name": "ANUJ KUMAR YADAV",
          "image_path": "data/faces/P001.jpg",
          "embedding": [0.12, -0.08, 0.34, 0.89, -0.21],
      },
      {
          "name": "AKSHAT SRIVASTAVA",
          "image_path": "data/faces/P002.jpg",
          "embedding": [0.31, 0.15, -0.12, 0.45, 0.67],
      },
      {
          "name": "HARSHIT PANDEY",
          "image_path": "data/faces/P003.jpg",
          "embedding": [0.08, -0.45, 0.88, -0.01, 0.11],
      },
      {
          "name": "ABHAY KUMAR",
          "image_path": "data/faces/P004.jpg",
          "embedding": [-0.22, 0.54, 0.19, 0.33, -0.41],
      },
      {
          "name": "PARUL NISHAD",
          "image_path": "data/faces/P005.jpg",
          "embedding": [0.71, -0.11, 0.05, 0.62, 0.18],
      },
      {
          "name": "ADITI MISHRA",
          "image_path": "data/faces/P006.jpg",
          "embedding": [-0.15, 0.28, -0.40, 0.51, 0.09],
      },
  ]

  clear_and_add_people(people_data)
