import asyncio
import json
import os
import socket
import ssl
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import av
import cv2
import numpy as np

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolo11n.pt"

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8443

# Laptop webcam
LAPTOP_CAMERA_INDEX = 0

# YOLO
CONFIDENCE = 0.25
IMAGE_SIZE = 640

# Process latest frame only
# 1 = maximum detection smoothness
# 2 = lower CPU/GPU load
PROCESS_EVERY_N_FRAMES = 1

# Bounding box smoothing
SMOOTHING_ALPHA = 0.25
STABILITY_DEADBAND = 3.0
MAX_BOX_STEP = 40.0

# Vehicle crop
VEHICLE_CROP_PADDING = 10
BEST_CROP_MIN_AREA = 1000

# Directories
ANPR_INPUT_DIR = "anpr_input"
EVENT_LOG_DIR = "events"

# SSL
CERT_FILE = "server_cert.pem"
KEY_FILE = "server_key.pem"

# Maximum phone cameras
MAX_PHONE_CAMERAS = 5

# MJPEG output
MJPEG_JPEG_QUALITY = 75
MJPEG_INTERVAL = 0.06


# ============================================================
# COCO CLASSES
# ============================================================

PERSON = 0
CAR = 2
MOTORCYCLE = 3
BUS = 5
TRUCK = 7

ALLOWED_CLASSES = {
    PERSON,
    CAR,
    MOTORCYCLE,
    BUS,
    TRUCK
}

CLASS_NAMES = {
    PERSON: "PERSON",
    CAR: "CAR",
    MOTORCYCLE: "MOTORCYCLE",
    BUS: "BUS",
    TRUCK: "TRUCK"
}

VEHICLE_CLASSES = {
    CAR,
    MOTORCYCLE,
    BUS,
    TRUCK
}


# ============================================================
# GLOBAL DATA
# ============================================================

sessions = {}

sessions_lock = threading.Lock()

phone_camera_counter = 0

global_event_counter = 0

global_event_logs = []

shutdown_event = threading.Event()

# ------------------------------------------------------------
# SHARED YOLO MODEL
# ------------------------------------------------------------
# IMPORTANT:
# Previously every camera session loaded its own YOLO model.
# With multiple cameras this can consume huge CPU/RAM/GPU.
# Now one model is shared by all cameras.
# ------------------------------------------------------------

shared_model = None

model_lock = threading.Lock()


def get_shared_model():

    global shared_model

    if shared_model is None:

        print(
            "\n[YOLO] Loading shared YOLO model..."
        )

        shared_model = YOLO(
            MODEL_PATH
        )

        print(
            "[YOLO] Shared YOLO model ready.\n"
        )

    return shared_model


# ============================================================
# UTILITY
# ============================================================

def get_local_ip():

    try:

        s = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM
        )

        s.connect(
            ("8.8.8.8", 80)
        )

        ip = s.getsockname()[0]

        s.close()

        return ip

    except Exception:

        return "127.0.0.1"


def ensure_directories():

    os.makedirs(
        ANPR_INPUT_DIR,
        exist_ok=True
    )

    os.makedirs(
        EVENT_LOG_DIR,
        exist_ok=True
    )


# ============================================================
# SSL CERTIFICATE
# ============================================================

def create_ssl_certificate():

    if (
        os.path.exists(CERT_FILE)
        and
        os.path.exists(KEY_FILE)
    ):

        print(
            "[SSL] Existing certificate found."
        )

        return

    print(
        "[SSL] Creating self-signed certificate..."
    )

    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        PrivateFormat,
        NoEncryption
    )

    import ipaddress

    local_ip = get_local_ip()

    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )

    subject = issuer = x509.Name([
        x509.NameAttribute(
            NameOID.COUNTRY_NAME,
            "IN"
        ),
        x509.NameAttribute(
            NameOID.ORGANIZATION_NAME,
            "IBVAP"
        ),
        x509.NameAttribute(
            NameOID.COMMON_NAME,
            local_ip
        )
    ])

    san_list = [
        x509.DNSName("localhost"),

        x509.IPAddress(
            ipaddress.IPv4Address(
                "127.0.0.1"
            )
        ),

        x509.IPAddress(
            ipaddress.IPv4Address(
                local_ip
            )
        )
    ]

    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(
            key.public_key()
        )
        .serial_number(
            x509.random_serial_number()
        )
        .not_valid_before(
            datetime.utcnow()
        )
        .not_valid_after(
            datetime.utcnow()
            + timedelta(days=365)
        )
        .add_extension(
            x509.SubjectAlternativeName(
                san_list
            ),
            critical=False
        )
        .sign(
            key,
            hashes.SHA256()
        )
    )

    with open(
        KEY_FILE,
        "wb"
    ) as f:

        f.write(
            key.private_bytes(
                Encoding.PEM,
                PrivateFormat.TraditionalOpenSSL,
                NoEncryption()
            )
        )

    with open(
        CERT_FILE,
        "wb"
    ) as f:

        f.write(
            certificate.public_bytes(
                Encoding.PEM
            )
        )

    print(
        "[SSL] Certificate created."
    )


# ============================================================
# CAMERA SESSION
# ============================================================

@dataclass
class CameraSession:

    camera_id: str

    source_type: str

    pc: object = None

    active: bool = True

    latest_frame: object = None

    processed_frame: object = None

    frame_lock: object = field(
        default_factory=threading.Lock
    )

    frame_number: int = 0

    fps: float = 0.0

    last_process_time: float = field(
        default_factory=time.time
    )

    processed_count: int = 0

    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------

    previous_boxes: dict = field(
        default_factory=dict
    )

    smooth_boxes: dict = field(
        default_factory=dict
    )

    # --------------------------------------------------------
    # Vehicle
    # --------------------------------------------------------

    vehicle_data: dict = field(
        default_factory=dict
    )

    best_vehicle_crops: dict = field(
        default_factory=dict
    )

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

    event_logs: list = field(
        default_factory=list
    )

    event_count: int = 0

    # --------------------------------------------------------
    # Async worker
    # --------------------------------------------------------

    worker_task: object = None

    # --------------------------------------------------------
    # Laptop capture
    # --------------------------------------------------------

    capture: object = None

    capture_thread: object = None

    capture_stop: object = None

    # --------------------------------------------------------
    # Camera dimensions
    # --------------------------------------------------------

    frame_width: int = 0

    frame_height: int = 0

    def initialize_model(self):

        # Shared model is used.
        get_shared_model()

    def camera_directory(self):

        path = os.path.join(
            ANPR_INPUT_DIR,
            self.camera_id
        )

        os.makedirs(
            path,
            exist_ok=True
        )

        return path

    def metadata_file(self):

        return os.path.join(
            self.camera_directory(),
            "vehicle_metadata.txt"
        )


# ============================================================
# CAMERA ID
# ============================================================

def get_new_phone_camera_id():

    global phone_camera_counter

    phone_camera_counter += 1

    return (
        f"CAM-{phone_camera_counter:02d}"
    )


# ============================================================
# EVENT LOGGING
# ============================================================

def log_event(
    session,
    event_type,
    severity,
    track_id=None,
    details=""
):

    global global_event_counter

    global_event_counter += 1

    session.event_count += 1

    event = {

        "event_id":
            global_event_counter,

        "timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "camera_id":
            session.camera_id,

        "event_type":
            event_type,

        "severity":
            severity,

        "track_id":
            track_id,

        "details":
            details
    }

    session.event_logs.append(
        event
    )

    global_event_logs.append(
        event
    )

    print(
        "\n"
        "==================================================\n"
        f"EVENT #{event['event_id']}\n"
        f"Camera   : {session.camera_id}\n"
        f"Type     : {event_type}\n"
        f"Severity : {severity}\n"
        f"Track ID : {track_id}\n"
        f"Details  : {details}\n"
        "==================================================\n"
    )

    return event


# ============================================================
# SAVE EVENTS
# ============================================================

def save_session_events(session):

    if not session.event_logs:

        return

    try:

        path = os.path.join(
            EVENT_LOG_DIR,
            f"{session.camera_id}_events.json"
        )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                session.event_logs,
                f,
                indent=4
            )

    except Exception as e:

        print(
            f"[EVENT] Save error: {e}"
        )


def save_all_events():

    try:

        path = os.path.join(
            EVENT_LOG_DIR,
            "all_events.json"
        )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                global_event_logs,
                f,
                indent=4
            )

        print(
            f"[EVENT] Saved: {path}"
        )

    except Exception as e:

        print(
            f"[EVENT] Global save error: {e}"
        )


# ============================================================
# VEHICLE METADATA
# ============================================================

def update_metadata_file(session):

    try:

        path = session.metadata_file()

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "IBVAP - VEHICLE ANALYTICS\n"
            )

            f.write(
                "====================================\n"
            )

            f.write(
                f"Camera ID: "
                f"{session.camera_id}\n"
            )

            f.write(
                f"Updated: "
                f"{datetime.now().isoformat(timespec='seconds')}\n"
            )

            f.write(
                "====================================\n\n"
            )

            for track_id, data in sorted(
                session.vehicle_data.items()
            ):

                f.write(
                    f"Track ID: {track_id}\n"
                )

                f.write(
                    f"Class: "
                    f"{data.get('class_name', '')}\n"
                )

                f.write(
                    f"Confidence: "
                    f"{data.get('confidence', 0):.2f}\n"
                )

                f.write(
                    f"Image: "
                    f"{data.get('filename', '')}\n"
                )

                f.write(
                    "------------------------------------\n"
                )

    except Exception as e:

        print(
            f"[ANPR] Metadata error: {e}"
        )


# ============================================================
# VEHICLE CROP
# ============================================================

def save_vehicle_crop(
    session,
    frame,
    track_id,
    box,
    class_name,
    confidence
):

    x1, y1, x2, y2 = box

    h, w = frame.shape[:2]

    x1 = max(
        0,
        int(x1) - VEHICLE_CROP_PADDING
    )

    y1 = max(
        0,
        int(y1) - VEHICLE_CROP_PADDING
    )

    x2 = min(
        w,
        int(x2) + VEHICLE_CROP_PADDING
    )

    y2 = min(
        h,
        int(y2) + VEHICLE_CROP_PADDING
    )

    if (
        x2 <= x1
        or
        y2 <= y1
    ):

        return

    crop = frame[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:

        return

    area = (
        crop.shape[0]
        *
        crop.shape[1]
    )

    if area < BEST_CROP_MIN_AREA:

        return

    old = (
        session.best_vehicle_crops.get(
            track_id
        )
    )

    old_area = (
        old["area"]
        if old is not None
        else 0
    )

    if area <= old_area:

        return

    filename = (
        f"vehicle_{track_id}.jpg"
    )

    path = os.path.join(
        session.camera_directory(),
        filename
    )

    try:

        cv2.imwrite(
            path,
            crop
        )

        session.best_vehicle_crops[
            track_id
        ] = {

            "area":
                area,

            "filename":
                filename,

            "path":
                path
        }

        session.vehicle_data[
            track_id
        ] = {

            "class_name":
                class_name,

            "confidence":
                float(confidence),

            "filename":
                filename,

            "area":
                area
        }

        update_metadata_file(
            session
        )

    except Exception as e:

        print(
            f"[ANPR] Crop save error: {e}"
        )


# ============================================================
# SAVE BEST VEHICLE CROPS
# ============================================================

def save_all_best_vehicle_crops(session):

    if not session.best_vehicle_crops:

        return

    print(
        f"[{session.camera_id}] "
        f"Vehicle crops saved: "
        f"{len(session.best_vehicle_crops)}"
    )

    update_metadata_file(
        session
    )


# ============================================================
# BOX SMOOTHING
# ============================================================

def smooth_box(
    session,
    track_id,
    box
):

    current = np.array(
        box,
        dtype=np.float32
    )

    previous = (
        session.smooth_boxes.get(
            track_id
        )
    )

    if previous is None:

        session.smooth_boxes[
            track_id
        ] = current

        return current.astype(int)

    diff = current - previous

    if np.all(
        np.abs(diff)
        <= STABILITY_DEADBAND
    ):

        return previous.astype(int)

    distance = np.linalg.norm(
        diff
    )

    if distance > MAX_BOX_STEP:

        ratio = (
            MAX_BOX_STEP / distance
        )

        current = (
            previous
            +
            diff * ratio
        )

    smoothed = (
        SMOOTHING_ALPHA * current
        +
        (1 - SMOOTHING_ALPHA)
        * previous
    )

    session.smooth_boxes[
        track_id
    ] = smoothed

    return smoothed.astype(int)


# ============================================================
# PROCESS FRAME
# ============================================================

def process_frame(
    session,
    frame
):

    session.initialize_model()

    output = frame.copy()

    h, w = output.shape[:2]

    try:

        model = get_shared_model()

        # ----------------------------------------------------
        # Shared model is protected by a lock.
        # This prevents multiple camera sessions from
        # simultaneously corrupting inference state.
        # ----------------------------------------------------

        with model_lock:

            results = model.track(

                source=frame,

                persist=True,

                tracker="bytetrack.yaml",

                conf=CONFIDENCE,

                imgsz=IMAGE_SIZE,

                verbose=False

            )

    except Exception as e:

        print(
            f"[{session.camera_id}] "
            f"YOLO error: {e}"
        )

        return output

    current_ids = set()

    vehicle_count = 0

    person_count = 0

    # ========================================================
    # YOLO RESULTS
    # ========================================================

    for result in results:

        if result.boxes is None:

            continue

        boxes = result.boxes

        for i in range(
            len(boxes)
        ):

            try:

                cls_id = int(
                    boxes.cls[i].item()
                )

                if cls_id not in ALLOWED_CLASSES:

                    continue

                confidence = float(
                    boxes.conf[i].item()
                )

                if confidence < CONFIDENCE:

                    continue

                raw_box = (
                    boxes.xyxy[
                        i
                    ]
                    .cpu()
                    .numpy()
                )

                x1, y1, x2, y2 = map(
                    float,
                    raw_box
                )

                # ------------------------------------------------
                # Track ID
                # ------------------------------------------------

                track_id = None

                if boxes.id is not None:

                    try:

                        track_id = int(
                            boxes.id[
                                i
                            ].item()
                        )

                    except Exception:

                        track_id = None

                if track_id is None:

                    track_id = i + 1

                current_ids.add(
                    track_id
                )

                # ------------------------------------------------
                # Smooth box
                # ------------------------------------------------

                box = smooth_box(

                    session,

                    track_id,

                    (
                        x1,
                        y1,
                        x2,
                        y2
                    )

                )

                sx1, sy1, sx2, sy2 = map(
                    int,
                    box
                )

                # Keep coordinates inside frame

                sx1 = max(
                    0,
                    min(
                        sx1,
                        w - 1
                    )
                )

                sy1 = max(
                    0,
                    min(
                        sy1,
                        h - 1
                    )
                )

                sx2 = max(
                    0,
                    min(
                        sx2,
                        w - 1
                    )
                )

                sy2 = max(
                    0,
                    min(
                        sy2,
                        h - 1
                    )
                )

                # ------------------------------------------------
                # Center
                # ------------------------------------------------

                cx = int(
                    (sx1 + sx2) / 2
                )

                cy = int(
                    (sy1 + sy2) / 2
                )

                # ------------------------------------------------
                # Counts
                # ------------------------------------------------

                if cls_id == PERSON:

                    person_count += 1

                elif cls_id in VEHICLE_CLASSES:

                    vehicle_count += 1

                    save_vehicle_crop(

                        session,

                        frame,

                        track_id,

                        (
                            sx1,
                            sy1,
                            sx2,
                            sy2
                        ),

                        CLASS_NAMES.get(
                            cls_id,
                            "VEHICLE"
                        ),

                        confidence

                    )

                # ------------------------------------------------
                # Color
                # ------------------------------------------------

                if cls_id == PERSON:

                    # RED person box
                    box_color = (
                        0,
                        0,
                        255
                    )

                else:

                    # BLUE vehicle box
                    box_color = (
                        255,
                        0,
                        0
                    )

                # ------------------------------------------------
                # Bounding box
                # ------------------------------------------------

                cv2.rectangle(

                    output,

                    (
                        sx1,
                        sy1
                    ),

                    (
                        sx2,
                        sy2
                    ),

                    box_color,

                    2

                )

                # ------------------------------------------------
                # Center point
                # ------------------------------------------------

                cv2.circle(

                    output,

                    (
                        cx,
                        cy
                    ),

                    5,

                    (
                        0,
                        255,
                        255
                    ),

                    -1

                )

                # ------------------------------------------------
                # Label
                # ------------------------------------------------

                label = (

                    f"{CLASS_NAMES.get}"
                    f"{(cls_id, 'OBJ')} "

                    f"ID:{track_id} "

                    f"{confidence:.2f}"

                )

                text_y = max(
                    25,
                    sy1 - 8
                )

                cv2.putText(

                    output,

                    label,

                    (
                        sx1,
                        text_y
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    box_color,

                    2

                )

                # ------------------------------------------------
                # Store previous box
                # ------------------------------------------------

                session.previous_boxes[
                    track_id
                ] = (

                    sx1,
                    sy1,
                    sx2,
                    sy2

                )

            except Exception as e:

                print(

                    f"[{session.camera_id}] "
                    f"Detection error: {e}"

                )

    # ========================================================
    # CLEANUP DISAPPEARED TRACKS
    # ========================================================

    old_ids = list(
        session.previous_boxes.keys()
    )

    for track_id in old_ids:

        if track_id not in current_ids:

            session.previous_boxes.pop(
                track_id,
                None
            )

            session.smooth_boxes.pop(
                track_id,
                None
            )

    # ========================================================
    # TOP BAR
    # ========================================================

    cv2.rectangle(

        output,

        (0, 0),

        (w, 70),

        (25, 25, 25),

        -1

    )

    cv2.putText(

        output,

        "IBVAP | BORDER SURVEILLANCE",

        (15, 28),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        (255, 255, 255),

        2

    )

    cv2.putText(

        output,

        f"CAMERA: {session.camera_id}",

        (15, 55),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        1

    )

    # ========================================================
    # FPS
    # ========================================================

    now = time.time()

    elapsed = (
        now
        -
        session.last_process_time
    )

    if elapsed > 0:

        instant_fps = (
            1.0 / elapsed
        )

        if session.fps <= 0:

            session.fps = instant_fps

        else:

            session.fps = (

                0.9
                *
                session.fps

                +

                0.1
                *
                instant_fps

            )

    session.last_process_time = now

    cv2.putText(

        output,

        f"FPS: {session.fps:.1f}",

        (
            max(
                10,
                w - 140
            ),
            30
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2

    )

    # ========================================================
    # ANALYTICS PANEL
    # ========================================================

    panel_x1 = 10
    panel_y1 = 85
    panel_x2 = 250
    panel_y2 = 185

    overlay = output.copy()

    cv2.rectangle(

        overlay,

        (
            panel_x1,
            panel_y1
        ),

        (
            min(
                panel_x2,
                w - 10
            ),
            min(
                panel_y2,
                h - 10
            )
        ),

        (20, 20, 20),

        -1

    )

    output = cv2.addWeighted(

        overlay,

        0.80,

        output,

        0.20,

        0

    )

    cv2.putText(

        output,

        f"PERSONS : {person_count}",

        (
            20,
            115
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        (255, 255, 255),

        2

    )

    cv2.putText(

        output,

        f"VEHICLES: {vehicle_count}",

        (
            20,
            145
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        (255, 255, 255),

        2

    )

    cv2.putText(

        output,

        f"EVENTS  : {session.event_count}",

        (
            20,
            175
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        (255, 255, 255),

        2

    )

    # ========================================================
    # SYSTEM STATUS
    # ========================================================

    cv2.putText(

        output,

        "SYSTEM STATUS: MONITORING",

        (
            20,
            max(
                25,
                h - 18
            )
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        (0, 255, 0),

        2

    )

    # ========================================================
    # UPDATE
    # ========================================================

    session.processed_count += 1

    session.frame_width = w
    session.frame_height = h

    return output


# ============================================================
# CAMERA PROCESSING WORKER
# ============================================================

async def camera_worker(session):

    print(
        f"[{session.camera_id}] "
        f"Processing worker started."
    )

    while session.active:

        frame = None

        # ----------------------------------------------------
        # Get ONLY latest frame.
        # Old frames are intentionally discarded.
        # ----------------------------------------------------

        with session.frame_lock:

            if session.latest_frame is not None:

                frame = (
                    session.latest_frame.copy()
                )

                # Clear the frame after taking it.
                # This prevents processing the same frame
                # again and again when camera is faster
                # than YOLO.
                session.latest_frame = None

        if frame is None:

            await asyncio.sleep(
                0.005
            )

            continue

        try:

            processed = await asyncio.to_thread(

                process_frame,

                session,

                frame

            )

            with session.frame_lock:

                session.processed_frame = processed

        except Exception as e:

            print(

                f"[{session.camera_id}] "
                f"Worker error: {e}"

            )

        await asyncio.sleep(
            0.001
        )

    print(
        f"[{session.camera_id}] "
        f"Processing worker stopped."
    )


# ============================================================
# PHONE WEBRTC TRACK READER
# ============================================================

async def read_phone_track(
    session,
    track
):

    print(
        f"[{session.camera_id}] "
        f"Phone video track started."
    )

    try:

        while session.active:

            frame = await track.recv()

            image = frame.to_ndarray(
                format="bgr24"
            )

            session.frame_number += 1

            if (
                session.frame_number
                %
                PROCESS_EVERY_N_FRAMES
                != 0
            ):

                continue

            h, w = image.shape[:2]

            session.frame_width = w
            session.frame_height = h

            # ------------------------------------------------
            # Latest frame only.
            # Never build a backlog.
            # ------------------------------------------------

            with session.frame_lock:

                session.latest_frame = image

    except Exception as e:

        print(

            f"[{session.camera_id}] "
            f"Phone track ended: {e}"

        )

    finally:

        session.active = False


# ============================================================
# LAPTOP CAMERA CAPTURE
# ============================================================

def laptop_capture_loop(
    session
):

    print(
        "[CAM-LAPTOP] "
        "Laptop webcam capture started."
    )

    cap = session.capture

    while (

        session.active

        and

        not session.capture_stop.is_set()

    ):

        ret, frame = cap.read()

        if not ret:

            time.sleep(
                0.02
            )

            continue

        session.frame_number += 1

        h, w = frame.shape[:2]

        session.frame_width = w
        session.frame_height = h

        with session.frame_lock:

            session.latest_frame = frame

        time.sleep(
            0.001
        )

    try:

        cap.release()

    except Exception:

        pass

    print(
        "[CAM-LAPTOP] "
        "Laptop webcam capture stopped."
    )


# ============================================================
# START LAPTOP CAMERA
# ============================================================

async def start_laptop_camera():

    with sessions_lock:

        if "CAM-LAPTOP" in sessions:

            return (

                False,

                "Laptop camera already running."

            )

        session = CameraSession(

            camera_id="CAM-LAPTOP",

            source_type="LAPTOP"

        )

        sessions[
            "CAM-LAPTOP"
        ] = session

    try:

        cap = cv2.VideoCapture(

            LAPTOP_CAMERA_INDEX,

            cv2.CAP_DSHOW

        )

        if not cap.isOpened():

            cap.release()

            with sessions_lock:

                sessions.pop(
                    "CAM-LAPTOP",
                    None
                )

            return (

                False,

                "Laptop webcam could not be opened."

            )

        # ----------------------------------------------------
        # Laptop camera remains 1280x720.
        # ----------------------------------------------------

        cap.set(

            cv2.CAP_PROP_FRAME_WIDTH,

            1280

        )

        cap.set(

            cv2.CAP_PROP_FRAME_HEIGHT,

            720

        )

        cap.set(

            cv2.CAP_PROP_FPS,

            30

        )

        session.capture = cap

        session.capture_stop = (
            threading.Event()
        )

        session.initialize_model()

        session.worker_task = (
            asyncio.create_task(

                camera_worker(
                    session
                )

            )
        )

        session.capture_thread = (
            threading.Thread(

                target=laptop_capture_loop,

                args=(session,),

                daemon=True

            )
        )

        session.capture_thread.start()

        print(
            "[CAM-LAPTOP] "
            "Laptop webcam ON."
        )

        return (

            True,

            "Laptop camera started."

        )

    except Exception as e:

        session.active = False

        with sessions_lock:

            sessions.pop(
                "CAM-LAPTOP",
                None
            )

        return False, str(e)


# ============================================================
# STOP LAPTOP CAMERA
# ============================================================

async def stop_laptop_camera():

    with sessions_lock:

        session = sessions.get(
            "CAM-LAPTOP"
        )

    if session is None:

        return (

            False,

            "Laptop camera is not running."

        )

    print(
        "[CAM-LAPTOP] "
        "Stopping..."
    )

    session.active = False

    if session.capture_stop:

        session.capture_stop.set()

    if session.capture_thread:

        session.capture_thread.join(
            timeout=2
        )

    if session.worker_task:

        try:

            await asyncio.wait_for(

                session.worker_task,

                timeout=2

            )

        except Exception:

            session.worker_task.cancel()

    save_session_events(
        session
    )

    save_all_best_vehicle_crops(
        session
    )

    with sessions_lock:

        sessions.pop(
            "CAM-LAPTOP",
            None
        )

    print(
        "[CAM-LAPTOP] "
        "Laptop webcam OFF."
    )

    return (

        True,

        "Laptop camera stopped."

    )


# ============================================================
# DASHBOARD HTML
# ============================================================

DASHBOARD_HTML = r"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta
name="viewport"
content="width=device-width, initial-scale=1.0"
>

<title>IBVAP - Border Surveillance</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: #0b0f14;
    color: white;
    font-family: Arial, sans-serif;
}

.header {
    background: #111820;
    padding: 18px 25px;
    border-bottom: 2px solid #263241;
}

.header h1 {
    margin: 0;
    font-size: 25px;
}

.header p {
    margin: 6px 0 0;
    color: #9da9b5;
}

.controls {
    padding: 15px 25px;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    background: #0f151c;
}

button {
    border: none;
    padding: 11px 18px;
    border-radius: 6px;
    cursor: pointer;
    font-weight: bold;
    color: white;
    background: #1677ff;
}

button:hover {
    opacity: 0.85;
}

.stop {
    background: #d92d20;
}

.status {
    padding: 10px 25px;
    color: #9da9b5;
}

.url-box {
    margin: 0 25px 15px;
    background: #141c25;
    padding: 15px;
    border-radius: 8px;
}

.url-box strong {
    color: #4da3ff;
}

.cameras {
    padding: 10px 25px 30px;

    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(420px, 1fr)
        );

    gap: 18px;
}

.camera-card {
    background: #111820;
    border: 1px solid #263241;
    border-radius: 10px;
    overflow: hidden;
}

.camera-title {
    padding: 12px 15px;
    font-size: 17px;
    font-weight: bold;
}

.camera-title span {
    color: #48d597;
}

/* ==========================================================
   IMPORTANT:
   Do NOT force a fixed height.
   This preserves the incoming camera's original ratio.
   ========================================================== */

.feed {
    width: 100%;
    height: auto;

    display: block;

    background: black;

    object-fit: contain;

    /* Browser should never stretch the image */
    max-width: 100%;

    /* Keep natural aspect ratio */
    aspect-ratio: auto;
}

.info {
    padding: 10px 15px;
    color: #aab5c1;
    font-size: 13px;
}

.no-camera {
    padding: 50px;
    text-align: center;
    color: #758191;
}

</style>

</head>

<body>

<div class="header">

<h1>
IBVAP | BORDER SURVEILLANCE
</h1>

<p>
Intelligent Border Video Analytics Platform
</p>

</div>


<div class="controls">

<button onclick="startLaptop()">
START LAPTOP CAMERA
</button>

<button
class="stop"
onclick="stopLaptop()"
>
STOP LAPTOP CAMERA
</button>

<button onclick="location.href='/phone'">
OPEN PHONE CAMERA PAGE
</button>

</div>


<div class="status" id="status">
Loading camera status...
</div>


<div class="url-box">

<strong>PHONE CAMERA URL:</strong>

<span id="phoneUrl">
Loading...
</span>

<br><br>

Open this URL on every phone connected to the same Wi-Fi.

</div>


<div
class="cameras"
id="cameras"
>

<div class="no-camera">
No cameras connected.
</div>

</div>


<script>

async function startLaptop() {

    try {

        const response =
            await fetch(
                "/api/laptop/start",
                {
                    method: "POST"
                }
            );

        const data =
            await response.json();

        alert(data.message);

        loadCameras();

    }

    catch (error) {

        alert(
            "Error: " + error
        );

    }

}


async function stopLaptop() {

    try {

        const response =
            await fetch(
                "/api/laptop/stop",
                {
                    method: "POST"
                }
            );

        const data =
            await response.json();

        alert(data.message);

        loadCameras();

    }

    catch (error) {

        alert(
            "Error: " + error
        );

    }

}


async function loadCameras() {

    try {

        const response =
            await fetch(
                "/api/cameras"
            );

        const data =
            await response.json();

        const container =
            document.getElementById(
                "cameras"
            );

        const cameras =
            data.cameras;

        document.getElementById(
            "status"
        ).innerText =
            "Connected cameras: "
            +
            cameras.length;

        if (
            cameras.length === 0
        ) {

            container.innerHTML =
                '<div class="no-camera">'
                +
                'No cameras connected.'
                +
                '</div>';

            return;

        }

        container.innerHTML = "";

        cameras.forEach(
            camera => {

                const card =
                    document.createElement(
                        "div"
                    );

                card.className =
                    "camera-card";

                card.innerHTML =

                    '<div class="camera-title">'
                    +
                    camera.camera_id
                    +
                    ' — '
                    +
                    '<span>'
                    +
                    camera.source_type
                    +
                    '</span>'
                    +
                    '</div>'
                    +

                    '<img '
                    +
                    'class="feed" '
                    +
                    'src="/mjpeg/'
                    +
                    camera.camera_id
                    +
                    '" '
                    +
                    'alt="Camera Feed">'
                    +

                    '<div class="info">'
                    +
                    'FPS: '
                    +
                    camera.fps.toFixed(1)
                    +
                    '<br>'
                    +
                    'Events: '
                    +
                    camera.events
                    +
                    '<br>'
                    +
                    'Processed Frames: '
                    +
                    camera.processed_frames
                    +
                    '<br>'
                    +
                    'Resolution: '
                    +
                    (
                        camera.width || "-"
                    )
                    +
                    ' × '
                    +
                    (
                        camera.height || "-"
                    )
                    +
                    '</div>';

                container.appendChild(
                    card
                );

            }
        );

    }

    catch (error) {

        console.log(error);

    }

}


async function loadPhoneUrl() {

    try {

        const response =
            await fetch(
                "/api/info"
            );

        const data =
            await response.json();

        document.getElementById(
            "phoneUrl"
        ).innerText =
            data.phone_url;

    }

    catch (error) {

        console.log(error);

    }

}


loadPhoneUrl();

loadCameras();

setInterval(
    loadCameras,
    1500
);

</script>

</body>

</html>
"""


# ============================================================
# PHONE HTML
# ============================================================

PHONE_HTML = r"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta
name="viewport"
content="width=device-width, initial-scale=1.0"
>

<title>IBVAP Phone Camera</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: #090d12;
    color: white;
    font-family: Arial, sans-serif;
    text-align: center;
}

h1 {
    padding: 20px 10px 5px;
    margin: 0;
}

p {
    color: #9da9b5;
}

/* ==========================================================
   IMPORTANT:
   Phone preview keeps its REAL camera aspect ratio.
   No forced 16:9 height.
   No stretching.
   ========================================================== */

.video-container {
    width: 95%;
    max-width: 900px;
    margin: 15px auto 0;

    display: flex;
    justify-content: center;
    align-items: center;

    background: black;
    border-radius: 10px;
    overflow: hidden;
}

video {
    display: block;

    width: 100%;
    height: auto;

    max-width: 100%;

    background: black;

    object-fit: contain;

    /* Preserve original stream ratio */
    aspect-ratio: auto;
}

button {
    margin: 15px 5px;
    padding: 12px 20px;
    border: none;
    border-radius: 7px;
    background: #1677ff;
    color: white;
    font-weight: bold;
}

.status {
    margin: 10px;
    padding: 12px;
    background: #151d26;
    border-radius: 8px;
}

.id {
    color: #4ddf9a;
    font-size: 20px;
    font-weight: bold;
}

.resolution {
    color: #8fa1b3;
    font-size: 13px;
    margin-top: 5px;
}

</style>

</head>

<body>

<h1>
IBVAP Phone Camera
</h1>

<p>
Border Surveillance Camera Input
</p>


<div class="status">

Status:

<span id="status">
Not connected
</span>

<br><br>

Camera ID:

<div
class="id"
id="cameraId"
>
-
</div>

<div
class="resolution"
id="resolution"
>
Resolution: -
</div>

</div>


<div class="video-container">

<video
id="video"
autoplay
playsinline
muted
>
</video>

</div>


<br>


<button
onclick="startCamera()"
>
START CAMERA
</button>


<button
onclick="stopCamera()"
>
STOP CAMERA
</button>


<script>

let localStream = null;

let pc = null;


function waitForIceGatheringComplete(
    peerConnection
) {

    return new Promise(
        resolve => {

            if (
                peerConnection.iceGatheringState
                ===
                "complete"
            ) {

                resolve();

                return;

            }

            function checkState() {

                if (
                    peerConnection.iceGatheringState
                    ===
                    "complete"
                ) {

                    peerConnection.removeEventListener(
                        "icegatheringstatechange",
                        checkState
                    );

                    resolve();

                }

            }

            peerConnection.addEventListener(
                "icegatheringstatechange",
                checkState
            );

        }
    );

}


async function startCamera() {

    try {

        document.getElementById(
            "status"
        ).innerText =
            "Requesting camera permission...";


        /*
         ======================================================
         IMPORTANT PHONE CAMERA FIX
         ======================================================

         We are NOT forcing:

             1280 x 720

         because that can make some phones crop/zoom.

         Instead:

         - environment = back camera preferred
         - resizeMode = none preferred
         - frame rate limited to 24 FPS
         - camera decides its native usable resolution
         ======================================================
        */

        localStream =
            await navigator.mediaDevices
            .getUserMedia({

                video: {

                    facingMode: {
                        ideal: "environment"
                    },

                    resizeMode: {
                        ideal: "none"
                    },

                    frameRate: {
                        ideal: 24,
                        max: 24
                    }

                },

                audio: false

            });


        const video =
            document.getElementById(
                "video"
            );


        video.srcObject =
            localStream;


        /*
         ======================================================
         GET ACTUAL CAMERA SETTINGS
         ======================================================
        */

        const videoTrack =
            localStream.getVideoTracks()[0];

        if (videoTrack) {

            const settings =
                videoTrack.getSettings();

            console.log(
                "Actual phone camera settings:",
                settings
            );

            if (
                settings.width
                &&
                settings.height
            ) {

                document.getElementById(
                    "resolution"
                ).innerText =
                    "Resolution: "
                    +
                    settings.width
                    +
                    " × "
                    +
                    settings.height;

            }

        }


        /*
         ======================================================
         WEBRTC
         ======================================================
        */

        pc =
            new RTCPeerConnection();


        localStream
            .getTracks()
            .forEach(
                track => {

                    pc.addTrack(
                        track,
                        localStream
                    );

                }
            );


        pc.onconnectionstatechange =
            () => {

                console.log(
                    "Connection:",
                    pc.connectionState
                );

                document.getElementById(
                    "status"
                ).innerText =
                    pc.connectionState;

            };


        const offer =
            await pc.createOffer();


        await pc.setLocalDescription(
            offer
        );


        await waitForIceGatheringComplete(
            pc
        );


        const response =
            await fetch(
                "/offer",
                {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({

                        sdp:
                            pc.localDescription.sdp,

                        type:
                            pc.localDescription.type

                    })

                }
            );


        if (!response.ok) {

            throw new Error(
                "Server rejected connection."
            );

        }


        const answer =
            await response.json();


        await pc.setRemoteDescription(
            answer
        );


        document.getElementById(
            "cameraId"
        ).innerText =
            answer.camera_id;


        document.getElementById(
            "status"
        ).innerText =
            "CONNECTED";

    }

    catch (error) {

        console.error(error);

        document.getElementById(
            "status"
        ).innerText =
            "ERROR: "
            +
            error.message;

        alert(
            "Camera connection failed:\n"
            +
            error.message
        );

    }

}


function stopCamera() {

    if (localStream) {

        localStream
            .getTracks()
            .forEach(
                track => track.stop()
            );

        localStream = null;

    }


    if (pc) {

        pc.close();

        pc = null;

    }


    document.getElementById(
        "video"
    ).srcObject =
        null;


    document.getElementById(
        "status"
    ).innerText =
        "STOPPED";


    document.getElementById(
        "cameraId"
    ).innerText =
        "-";


    document.getElementById(
        "resolution"
    ).innerText =
        "Resolution: -";

}

</script>

</body>

</html>
"""


# ============================================================
# DASHBOARD ROUTE
# ============================================================

async def dashboard_handler(
    request
):

    return web.Response(

        text=DASHBOARD_HTML,

        content_type="text/html"

    )


# ============================================================
# PHONE ROUTE
# ============================================================

async def phone_handler(
    request
):

    return web.Response(

        text=PHONE_HTML,

        content_type="text/html"

    )


# ============================================================
# API INFO
# ============================================================

async def info_handler(
    request
):

    local_ip = get_local_ip()

    return web.json_response({

        "name":
            "IBVAP",

        "phone_url":
            f"https://{local_ip}:{SERVER_PORT}/phone",

        "dashboard_url":
            f"https://{local_ip}:{SERVER_PORT}/",

        "local_ip":
            local_ip,

        "port":
            SERVER_PORT

    })


# ============================================================
# API CAMERAS
# ============================================================

async def cameras_handler(
    request
):

    result = []

    with sessions_lock:

        current_sessions = list(
            sessions.values()
        )

    for session in current_sessions:

        result.append({

            "camera_id":
                session.camera_id,

            "source_type":
                session.source_type,

            "active":
                session.active,

            "fps":
                session.fps,

            "events":
                session.event_count,

            "processed_frames":
                session.processed_count,

            "width":
                session.frame_width,

            "height":
                session.frame_height

        })

    return web.json_response({

        "cameras":
            result

    })


# ============================================================
# LAPTOP START API
# ============================================================

async def laptop_start_handler(
    request
):

    success, message = (
        await start_laptop_camera()
    )

    return web.json_response({

        "success":
            success,

        "message":
            message

    })


# ============================================================
# LAPTOP STOP API
# ============================================================

async def laptop_stop_handler(
    request
):

    success, message = (
        await stop_laptop_camera()
    )

    return web.json_response({

        "success":
            success,

        "message":
            message

    })


# ============================================================
# WEBRTC OFFER
# ============================================================

async def offer_handler(
    request
):

    try:

        params = await request.json()

        offer = RTCSessionDescription(

            sdp=params["sdp"],

            type=params["type"]

        )

    except Exception as e:

        return web.json_response(

            {
                "error":
                    f"Invalid offer: {e}"
            },

            status=400

        )

    # --------------------------------------------------------
    # Camera limit
    # --------------------------------------------------------

    with sessions_lock:

        phone_count = len([

            s

            for s in sessions.values()

            if s.source_type == "PHONE"

        ])

    if phone_count >= MAX_PHONE_CAMERAS:

        return web.json_response(

            {
                "error":
                    "Maximum phone camera limit reached."
            },

            status=503

        )

    # --------------------------------------------------------
    # New camera
    # --------------------------------------------------------

    camera_id = (
        get_new_phone_camera_id()
    )

    session = CameraSession(

        camera_id=camera_id,

        source_type="PHONE"

    )

    pc = RTCPeerConnection()

    session.pc = pc

    with sessions_lock:

        sessions[
            camera_id
        ] = session

    print(

        f"[{camera_id}] "
        f"New phone connection."

    )

    # --------------------------------------------------------
    # Track handler
    # --------------------------------------------------------

    @pc.on("track")
    def on_track(track):

        print(

            f"[{camera_id}] "
            f"Received track: "
            f"{track.kind}"

        )

        if track.kind == "video":

            asyncio.create_task(

                read_phone_track(

                    session,

                    track

                )

            )

    # --------------------------------------------------------
    # Connection state
    # --------------------------------------------------------

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():

        state = pc.connectionState

        print(

            f"[{camera_id}] "
            f"Connection state: {state}"

        )

        if state in (

            "failed",

            "closed",

            "disconnected"

        ):

            await cleanup_phone_session(
                camera_id
            )

    try:

        await pc.setRemoteDescription(
            offer
        )

        answer = await pc.createAnswer()

        await pc.setLocalDescription(
            answer
        )

        # Start processing worker

        session.worker_task = (
            asyncio.create_task(

                camera_worker(
                    session
                )

            )
        )

        return web.json_response({

            "sdp":
                pc.localDescription.sdp,

            "type":
                pc.localDescription.type,

            "camera_id":
                camera_id

        })

    except Exception as e:

        print(

            f"[{camera_id}] "
            f"WebRTC error: {e}"

        )

        await cleanup_phone_session(
            camera_id
        )

        return web.json_response(

            {
                "error":
                    str(e)
            },

            status=500

        )


# ============================================================
# CLEANUP PHONE
# ============================================================

async def cleanup_phone_session(
    camera_id
):

    with sessions_lock:

        session = sessions.get(
            camera_id
        )

    if session is None:

        return

    if not session.active:

        return

    print(

        f"[{camera_id}] "
        f"Cleaning up..."

    )

    session.active = False

    # --------------------------------------------------------
    # Close WebRTC
    # --------------------------------------------------------

    if session.pc:

        try:

            await session.pc.close()

        except Exception:

            pass

    # --------------------------------------------------------
    # Worker
    # --------------------------------------------------------

    current_task = (
        asyncio.current_task()
    )

    if (

        session.worker_task

        and

        session.worker_task != current_task

    ):

        try:

            await asyncio.wait_for(

                session.worker_task,

                timeout=2

            )

        except Exception:

            session.worker_task.cancel()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_session_events(
        session
    )

    save_all_best_vehicle_crops(
        session
    )

    # --------------------------------------------------------
    # Remove
    # --------------------------------------------------------

    with sessions_lock:

        sessions.pop(
            camera_id,
            None
        )

    print(

        f"[{camera_id}] "
        f"Disconnected."

    )


# ============================================================
# MJPEG STREAM
# ============================================================

async def mjpeg_handler(
    request
):

    camera_id = request.match_info[
        "camera_id"
    ]

    with sessions_lock:

        session = sessions.get(
            camera_id
        )

    if session is None:

        return web.Response(

            status=404,

            text="Camera not found."

        )

    response = web.StreamResponse(

        status=200,

        headers={

            "Content-Type":
                "multipart/x-mixed-replace; boundary=frame",

            "Cache-Control":
                "no-cache",

            "Pragma":
                "no-cache",

            "Connection":
                "close"

        }

    )

    await response.prepare(
        request
    )

    try:

        while session.active:

            frame = None

            with session.frame_lock:

                if (
                    session.processed_frame
                    is not None
                ):

                    frame = (
                        session.processed_frame.copy()
                    )

            if frame is not None:

                ok, jpeg = cv2.imencode(

                    ".jpg",

                    frame,

                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        MJPEG_JPEG_QUALITY
                    ]

                )

                if ok:

                    data = jpeg.tobytes()

                    await response.write(

                        b"--frame\r\n"

                        b"Content-Type: "
                        b"image/jpeg\r\n"

                        +

                        (
                            f"Content-Length: "
                            f"{len(data)}\r\n\r\n"
                        ).encode()

                        +

                        data

                        +

                        b"\r\n"

                    )

            await asyncio.sleep(
                MJPEG_INTERVAL
            )

    except (

        ConnectionResetError,

        BrokenPipeError,

        asyncio.CancelledError

    ):

        pass

    except Exception as e:

        print(

            f"[MJPEG] "
            f"{camera_id}: {e}"

        )

    return response


# ============================================================
# EVENTS API
# ============================================================

async def events_handler(
    request
):

    return web.json_response({

        "events":
            global_event_logs

    })


# ============================================================
# SHUTDOWN
# ============================================================

async def shutdown_server(
    app
):

    print(

        "\n[SERVER] "
        "Shutting down..."

    )

    shutdown_event.set()

    # --------------------------------------------------------
    # Stop laptop
    # --------------------------------------------------------

    try:

        await stop_laptop_camera()

    except Exception:

        pass

    # --------------------------------------------------------
    # Stop phones
    # --------------------------------------------------------

    with sessions_lock:

        phone_ids = [

            camera_id

            for camera_id, session
            in sessions.items()

            if session.source_type == "PHONE"

        ]

    for camera_id in phone_ids:

        try:

            await cleanup_phone_session(
                camera_id
            )

        except Exception:

            pass

    save_all_events()

    print(

        "[SERVER] "
        "Shutdown complete."

    )


# ============================================================
# CREATE APP
# ============================================================

def create_app():

    app = web.Application()

    app.router.add_get(
        "/",
        dashboard_handler
    )

    app.router.add_get(
        "/phone",
        phone_handler
    )

    app.router.add_get(
        "/api/info",
        info_handler
    )

    app.router.add_get(
        "/api/cameras",
        cameras_handler
    )

    app.router.add_get(
        "/api/events",
        events_handler
    )

    app.router.add_post(
        "/offer",
        offer_handler
    )

    app.router.add_post(
        "/api/laptop/start",
        laptop_start_handler
    )

    app.router.add_post(
        "/api/laptop/stop",
        laptop_stop_handler
    )

    app.router.add_get(
        "/mjpeg/{camera_id}",
        mjpeg_handler
    )

    app.on_cleanup.append(
        shutdown_server
    )

    return app


# ============================================================
# MAIN
# ============================================================

def main():

    print(

        "\n"
        "============================================================\n"
        "       IBVAP - INTELLIGENT BORDER VIDEO ANALYTICS\n"
        "============================================================\n"

    )

    ensure_directories()

    # --------------------------------------------------------
    # Check YOLO model
    # --------------------------------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        print(
            f"\nERROR: {MODEL_PATH} not found.\n"
        )

        print(

            "Place yolo11n.pt in the same folder "
            "as main.py."

        )

        return

    # --------------------------------------------------------
    # Load shared YOLO model once
    # --------------------------------------------------------

    try:

        get_shared_model()

    except Exception as e:

        print(
            "\nYOLO model loading failed:"
        )

        print(e)

        return

    # --------------------------------------------------------
    # SSL
    # --------------------------------------------------------

    try:

        create_ssl_certificate()

    except Exception as e:

        print(
            "\nSSL certificate creation failed:"
        )

        print(e)

        print(
            "\nInstall cryptography using:"
        )

        print(
            "py -3.12 -m pip install cryptography"
        )

        return

    # --------------------------------------------------------
    # SSL context
    # --------------------------------------------------------

    ssl_context = ssl.create_default_context(

        ssl.Purpose.CLIENT_AUTH

    )

    ssl_context.load_cert_chain(

        CERT_FILE,

        KEY_FILE

    )

    # --------------------------------------------------------
    # URLs
    # --------------------------------------------------------

    local_ip = get_local_ip()

    dashboard_url = (

        f"https://{local_ip}:{SERVER_PORT}/"

    )

    phone_url = (

        f"https://{local_ip}:{SERVER_PORT}/phone"

    )

    print(

        "\n============================================================"

    )

    print(
        "\nDASHBOARD:"
    )

    print(
        dashboard_url
    )

    print(
        "\nPHONE CAMERA:"
    )

    print(
        phone_url
    )

    print(

        "\n============================================================"

    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "1. Laptop and phones should be on same Wi-Fi/LAN."
    )

    print(
        "2. Open PHONE CAMERA URL on each phone."
    )

    print(
        "3. Allow camera permission."
    )

    print(
        "4. Open DASHBOARD URL on laptop."
    )

    print(
        "5. Use START LAPTOP CAMERA when you want laptop webcam."
    )

    print(
        "6. Phone camera keeps its original aspect ratio."
    )

    print(
        "7. Multiple cameras use one shared YOLO model."
    )

    print(

        "\n============================================================\n"

    )

    # --------------------------------------------------------
    # Create app
    # --------------------------------------------------------

    app = create_app()

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    web.run_app(

        app,

        host=SERVER_HOST,

        port=SERVER_PORT,

        ssl_context=ssl_context

    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()