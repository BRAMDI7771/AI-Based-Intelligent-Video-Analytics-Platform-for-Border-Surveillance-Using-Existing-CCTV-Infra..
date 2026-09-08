import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os
import socket
import sqlite3
import ssl
import threading
import time
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
import av
import cv2
import numpy as np
from ultralytics import YOLO

#
# CONFIGURATION & PATHS
#
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "app.db")

MODEL_PATH = "yolo11n.pt"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8443
LAPTOP_CAMERA_INDEX = 1

CONFIDENCE = 0.25
IMAGE_SIZE = 640
PROCESS_EVERY_N_FRAMES = 1

ANPR_INPUT_DIR = "anpr_input"
EVENT_LOG_DIR = "events"
CERT_FILE = "server_cert.pem"
KEY_FILE = "server_key.pem"
MAX_PHONE_CAMERAS = 5
MJPEG_JPEG_QUALITY = 75
MJPEG_INTERVAL = 0.06

#
# GLOBAL DATA
#
sessions = {}
sessions_lock = threading.Lock()
phone_camera_counter = 0
global_event_counter = 0
global_event_logs = []
shutdown_event = threading.Event()

shared_model = None
model_lock = threading.Lock()


#
# FACE RECOGNITION UTILITIES (Database & Vector Similarity)
#
def cosine_similarity(v1, v2):
  a = np.array(v1)
  b = np.array(v2)
  return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def match_face(input_embedding, threshold=0.6):
  if not os.path.exists(DB_PATH):
    return {
        "id": "UNKNOWN",
        "name": "DB Not Found",
        "score": 0.0,
        "status": "NO MATCH",
    }

  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()
  cursor.execute("SELECT person_id, name, image_path, embedding FROM persons")
  records = cursor.fetchall()
  conn.close()

  best_match = None
  highest_score = -1.0

  for person_id, name, img_path, embed_str in records:
    try:
      stored_embedding = json.loads(embed_str)
      score = cosine_similarity(input_embedding, stored_embedding)
      if score > highest_score:
        highest_score = score
        best_match = {
            "id": f"P{person_id:03d}",
            "name": name,
            "score": round(score * 100, 1),
        }
    except Exception:
      continue

  if best_match and highest_score >= threshold:
    best_match["status"] = "MATCH"
  else:
    best_match = {
        "id": "UNKNOWN",
        "name": "Unknown",
        "score": round(highest_score * 100, 1) if best_match else 0.0,
        "status": "NO MATCH",
    }
  return best_match


def get_shared_model():
  global shared_model
  if shared_model is None:
    print("\n[YOLO] Loading shared YOLO model...")
    shared_model = YOLO(MODEL_PATH)
    print("[YOLO] Shared YOLO model ready.\n")
  return shared_model


def get_local_ip():
  try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip
  except Exception:
    return "127.0.0.1"


def ensure_directories():
  os.makedirs(ANPR_INPUT_DIR, exist_ok=True)
  os.makedirs(EVENT_LOG_DIR, exist_ok=True)


def create_ssl_certificate():
  if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
    print("[SSL] Existing certificate found.")
    return
  print("[SSL] Creating self-signed certificate...")
  from cryptography import x509
  from cryptography.hazmat.primitives import hashes
  from cryptography.hazmat.primitives.asymmetric import rsa
  from cryptography.hazmat.primitives.serialization import (
      Encoding,
      NoEncryption,
      PrivateFormat,
  )
  from cryptography.x509.oid import NameOID
  import ipaddress

  local_ip = get_local_ip()
  key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
  subject = issuer = x509.Name([
      x509.NameAttribute(NameOID.COUNTRY_NAME, "IN"),
      x509.NameAttribute(NameOID.ORGANIZATION_NAME, "IBVAP"),
      x509.NameAttribute(NameOID.COMMON_NAME, local_ip),
  ])
  san_list = [
      x509.DNSName("localhost"),
      x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
      x509.IPAddress(ipaddress.IPv4Address(local_ip)),
  ]
  certificate = (
      x509.CertificateBuilder()
      .subject_name(subject)
      .issuer_name(issuer)
      .public_key(key.public_key())
      .serial_number(x509.random_serial_number())
      .not_valid_before(datetime.utcnow())
      .not_valid_after(datetime.utcnow() + timedelta(days=365))
      .add_extension(
          x509.SubjectAlternativeName(san_list), critical=False
      )
      .sign(key, hashes.SHA256())
  )
  with open(KEY_FILE, "wb") as f:
    f.write(
        key.private_bytes(
            Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()
        )
    )
  with open(CERT_FILE, "wb") as f:
    f.write(certificate.public_bytes(Encoding.PEM))
  print("[SSL] Certificate created.")


@dataclass
class CameraSession:
  camera_id: str
  source_type: str
  pc: object = None
  active: bool = True
  latest_frame: object = None
  processed_frame: object = None
  frame_lock: object = field(default_factory=threading.Lock)
  frame_number: int = 0
  fps: float = 0.0
  last_process_time: float = field(default_factory=time.time)
  processed_count: int = 0
  event_logs: list = field(default_factory=list)
  event_count: int = 0
  worker_task: object = None
  capture: object = None
  capture_thread: object = None
  capture_stop: object = None
  frame_width: int = 0
  frame_height: int = 0


def get_new_phone_camera_id():
  global phone_camera_counter
  phone_camera_counter += 1
  return f"CAM-{phone_camera_counter:02d}"


#
# CORE FRAME PROCESSING (YOLO + Face Recognition Dashboard Integration)
#
def process_frame(session, frame):
  output = frame.copy()
  h, w = output.shape[:2]

  # Face detection Haar Cascade initialization
  face_cascade = cv2.CascadeClassifier(
      cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
  )
  gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(
      gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
  )

  person_count = len(faces)

  for x, y, fw, fh in faces:
    # MVP Simulated Vector (Matching registered P001 profile)
    simulated_vector = [0.12, -0.08, 0.34, 0.89, -0.21]
    match_data = match_face(simulated_vector)

    color = (0, 255, 0) if match_data["status"] == "MATCH" else (0, 0, 255)

    # Face Bounding Box
    cv2.rectangle(output, (x, y), (x + fw, y + fh), color, 2)

    # Top-Left Intelligence Overlay Card
    cv2.rectangle(output, (10, 80), (330, 215), (20, 20, 20), -1)
    cv2.rectangle(output, (10, 80), (330, 215), color, 2)

    cv2.putText(
        output,
        "PERSON DETECTED",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        output,
        f"ID    : {match_data['id']}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )
    cv2.putText(
        output,
        f"Name  : {match_data['name']}",
        (20, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )
    cv2.putText(
        output,
        f"Score : {match_data['score']}%",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )
    cv2.putText(
        output,
        f"Status: {match_data['status']}",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )

  # TOP SURVEILLANCE HEADER
  cv2.rectangle(output, (0, 0), (w, 70), (25, 25, 25), -1)
  cv2.putText(
      output,
      "IBVAP | BORDER SURVEILLANCE MVP",
      (15, 28),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.75,
      (255, 255, 255),
      2,
  )
  cv2.putText(
      output,
      f"CAMERA: {session.camera_id}",
      (15, 55),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.55,
      (255, 255, 255),
      1,
  )

  # FPS CALCULATION
  now = time.time()
  elapsed = now - session.last_process_time
  if elapsed > 0:
    instant_fps = 1.0 / elapsed
    session.fps = (
        instant_fps
        if session.fps <= 0
        else (0.9 * session.fps + 0.1 * instant_fps)
    )
  session.last_process_time = now

  cv2.putText(
      output,
      f"FPS: {session.fps:.1f}",
      (max(10, w - 140), 30),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.65,
      (255, 255, 255),
      2,
  )

  session.processed_count += 1
  session.frame_width = w
  session.frame_height = h
  return output


async def camera_worker(session):
  while session.active:
    frame = None
    with session.frame_lock:
      if session.latest_frame is not None:
        frame = session.latest_frame.copy()
        session.latest_frame = None

    if frame is None:
      await asyncio.sleep(0.005)
      continue

    try:
      processed = await asyncio.to_thread(process_frame, session, frame)
      with session.frame_lock:
        session.processed_frame = processed
    except Exception as e:
      print(f"[{session.camera_id}] Worker error: {e}")

    await asyncio.sleep(0.001)


async def read_phone_track(session, track):
  try:
    while session.active:
      frame = await track.recv()
      image = frame.to_ndarray(format="bgr24")
      session.frame_number += 1
      if session.frame_number % PROCESS_EVERY_N_FRAMES != 0:
        continue
      with session.frame_lock:
        session.latest_frame = image
  except Exception as e:
    pass
  finally:
    session.active = False


def laptop_capture_loop(session):
  cap = session.capture
  while session.active and not session.capture_stop.is_set():
    ret, frame = cap.read()
    if not ret:
      time.sleep(0.02)
      continue
    session.frame_number += 1
    with session.frame_lock:
      session.latest_frame = frame
    time.sleep(0.001)
  try:
    cap.release()
  except Exception:
    pass


async def start_laptop_camera():
  with sessions_lock:
    if "CAM-LAPTOP" in sessions:
      return False, "Laptop camera already running."

    session = CameraSession(
        camera_id="CAM-LAPTOP", source_type="LAPTOP"
    )
    sessions["CAM-LAPTOP"] = session

  try:
    cap = cv2.VideoCapture(LAPTOP_CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
      cap.release()
      with sessions_lock:
        sessions.pop("CAM-LAPTOP", None)
      return False, "Laptop webcam could not be opened."

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    session.capture = cap
    session.capture_stop = threading.Event()
    session.worker_task = asyncio.create_task(camera_worker(session))
    session.capture_thread = threading.Thread(
        target=laptop_capture_loop, args=(session,), daemon=True
    )
    session.capture_thread.start()
    return True, "Laptop camera started."
  except Exception as e:
    session.active = False
    with sessions_lock:
      sessions.pop("CAM-LAPTOP", None)
    return False, str(e)


async def stop_laptop_camera():
  with sessions_lock:
    session = sessions.get("CAM-LAPTOP")
    if session is None:
      return False, "Laptop camera is not running."

  session.active = False
  if session.capture_stop:
    session.capture_stop.set()
  if session.capture_thread:
    session.capture_thread.join(timeout=2)
  if session.worker_task:
    try:
      await asyncio.wait_for(session.worker_task, timeout=2)
    except Exception:
      session.worker_task.cancel()

  with sessions_lock:
    sessions.pop("CAM-LAPTOP", None)

  return True, "Laptop camera stopped."


#
# WEB DASHBOARD TEMPLATES & HANDLERS
#
DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>IBVAP - Border Surveillance</title>
    <style>
        body { margin: 0; background: #0b0f14; color: white; font-family: Arial, sans-serif; }
        .header { background: #111820; padding: 18px 25px; border-bottom: 2px solid #263241; }
        .controls { padding: 15px 25px; display: flex; gap: 10px; background: #0f151c; }
        button { border: none; padding: 11px 18px; border-radius: 6px; cursor: pointer; font-weight: bold; color: white; background: #1677ff; }
        .stop { background: #d92d20; }
        .cameras { padding: 20px; display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }
        .camera-card { background: #111820; border: 1px solid #263241; border-radius: 10px; overflow: hidden; }
        .feed { width: 100%; height: auto; display: block; background: black; }
    </style>
</head>
<body>
    <div class="header">
        <h1>IBVAP BORDER SURVEILLANCE</h1>
        <p>Intelligent Multi-Camera Face Matching Engine</p>
    </div>
    <div class="controls">
        <button onclick="startLaptop()">START LAPTOP CAMERA</button>
        <button class="stop" onclick="stopLaptop()">STOP LAPTOP CAMERA</button>
        <button onclick="location.href='/phone'">OPEN PHONE CAMERA PAGE</button>
    </div>
    <div class="cameras" id="cameras"></div>
    <script>
        async function startLaptop() { await fetch("/api/laptop/start", {method: "POST"}); loadCameras(); }
        async function stopLaptop() { await fetch("/api/laptop/stop", {method: "POST"}); loadCameras(); }
        async function loadCameras() {
            const res = await fetch("/api/cameras");
            const data = await res.json();
            const container = document.getElementById("cameras");
            container.innerHTML = "";
            data.cameras.forEach(cam => {
                container.innerHTML += `<div class="camera-card">
                    <div style="padding:10px;">${cam.camera_id} (${cam.source_type})</div>
                    <img class="feed" src="/mjpeg/${cam.camera_id}">
                </div>`;
            });
        }
        setInterval(loadCameras, 2000);
        loadCameras();
    </script>
</body>
</html>
"""

PHONE_HTML = r"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phone Surveillance Stream</title>
    <style>
        body { margin: 0; background: #090d12; color: white; text-align: center; font-family: Arial; }
        video { width: 90%; max-width: 600px; margin-top: 15px; border-radius: 8px; background: black; }
        button { margin: 15px; padding: 12px 20px; border-radius: 6px; border: none; background: #1677ff; color: white; font-weight: bold; }
    </style>
</head>
<body>
    <h2>Phone Camera Stream</h2>
    <video id="video" autoplay playsinline muted></video><br>
    <button onclick="startCamera()">START STREAM</button>
    <script>
        let localStream, pc;
        async function startCamera() {
            localStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } }, audio: false });
            document.getElementById("video").srcObject = localStream;
            pc = new RTCPeerConnection();
            localStream.getTracks().forEach(track => pc.addTrack(track, localStream));
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            
            const response = await fetch("/offer", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type })
            });
            const answer = await response.json();
            await pc.setRemoteDescription(answer);
        }
    </script>
</body>
</html>
"""


async def dashboard_handler(request):
  return web.Response(text=DASHBOARD_HTML, content_type="text/html")


async def phone_handler(request):
  return web.Response(text=PHONE_HTML, content_type="text/html")


async def cameras_handler(request):
  result = []
  with sessions_lock:
    for session in sessions.values():
      result.append({
          "camera_id": session.camera_id,
          "source_type": session.source_type,
          "active": session.active,
          "fps": session.fps,
      })
  return web.json_response({"cameras": result})


async def laptop_start_handler(request):
  s, m = await start_laptop_camera()
  return web.json_response({"success": s, "message": m})


async def laptop_stop_handler(request):
  s, m = await stop_laptop_camera()
  return web.json_response({"success": s, "message": m})


async def offer_handler(request):
  params = await request.json()
  offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

  camera_id = get_new_phone_camera_id()
  session = CameraSession(camera_id=camera_id, source_type="PHONE")
  pc = RTCPeerConnection()
  session.pc = pc

  with sessions_lock:
    sessions[camera_id] = session

  @pc.on("track")
  def on_track(track):
    if track.kind == "video":
      asyncio.create_task(read_phone_track(session, track))

  await pc.setRemoteDescription(offer)
  answer = await pc.createAnswer()
  await pc.setLocalDescription(answer)

  session.worker_task = asyncio.create_task(camera_worker(session))
  return web.json_response({
      "sdp": pc.localDescription.sdp,
      "type": pc.localDescription.type,
      "camera_id": camera_id,
  })


async def mjpeg_handler(request):
  camera_id = request.match_info["camera_id"]
  with sessions_lock:
    session = sessions.get(camera_id)

  if not session:
    return web.Response(status=404, text="Camera not found.")

  response = web.StreamResponse(
      status=200,
      headers={
          "Content-Type": "multipart/x-mixed-replace; boundary=frame",
          "Cache-Control": "no-cache",
      },
  )
  await response.prepare(request)

  try:
    while session.active:
      frame = None
      with session.frame_lock:
        if session.processed_frame is not None:
          frame = session.processed_frame.copy()

      if frame is not None:
        ok, jpeg = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_JPEG_QUALITY]
        )
        if ok:
          data = jpeg.tobytes()
          await response.write(
              b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
              + str(len(data)).encode()
              + b"\r\n\r\n"
              + data
              + b"\r\n"
          )
      await asyncio.sleep(MJPEG_INTERVAL)
  except Exception:
    pass
  return response


def create_app():
  app = web.Application()
  app.router.add_get("/", dashboard_handler)
  app.router.add_get("/phone", phone_handler)
  app.router.add_get("/api/cameras", cameras_handler)
  app.router.add_post("/offer", offer_handler)
  app.router.add_post("/api/laptop/start", laptop_start_handler)
  app.router.add_post("/api/laptop/stop", laptop_stop_handler)
  app.router.add_get("/mjpeg/{camera_id}", mjpeg_handler)
  return app


def main():
  ensure_directories()
  try:
    create_ssl_certificate()
  except Exception:
    pass

  ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
  ssl_context.load_cert_chain(CERT_FILE, KEY_FILE)

  local_ip = get_local_ip()
  print(f"\n==========================================")
  print(f"DASHBOARD URL : https://{local_ip}:{SERVER_PORT}/")
  print(f"PHONE CAM URL : https://{local_ip}:{SERVER_PORT}/phone")
  print(f"==========================================\n")

  app = create_app()
  web.run_app(
      app, host=SERVER_HOST, port=SERVER_PORT, ssl_context=ssl_context
  )


if __name__ == "__main__":
  main()
