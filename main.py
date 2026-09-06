import asyncio
import json
import os
from dotenv import load_dotenv
load_dotenv()
import re
import shutil
import socket
import ssl
import subprocess
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import av
import cv2
import numpy as np

from aiohttp import web

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
)

from ultralytics import YOLO


# ============================================================
# IBVAP - INTELLIGENT BORDER VIDEO ANALYTICS
# STABLE PERSON + VEHICLE DETECTION VERSION
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolo11n.pt"

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8443

LAPTOP_CAMERA_INDEX = 0


# ============================================================
# CLOUDFLARE QUICK TUNNEL
# ============================================================

# cloudflared.exe location on this laptop.
CLOUDFLARED_PATH = r"C:\cloudflared\cloudflared.exe"

# Automatically start Cloudflare when main.py starts.
AUTO_START_CLOUDFLARE = True


# ============================================================
# YOLO SETTINGS
# ============================================================

# Increased from 0.25 to reduce false detections.
CONFIDENCE = 0.40

IMAGE_SIZE = 640

# Process every frame.
PROCESS_EVERY_N_FRAMES = 1


# ============================================================
# PERSON FILTERING
# ============================================================

# A person detection must be confirmed for this many
# consecutive frames before being displayed.
PERSON_CONFIRM_FRAMES = 2

# Minimum person box height.
# This removes very tiny false "person" detections.
MIN_PERSON_HEIGHT = 55

# Minimum person box area.
MIN_PERSON_AREA = 1200

# If a small suspected person is mostly inside another
# person box, it will be rejected.
PERSON_CONTAINMENT_THRESHOLD = 0.60

# Minimum confidence specifically required for a person.
PERSON_MIN_CONFIDENCE = 0.40


# ============================================================
# VEHICLE FILTERING
# ============================================================

VEHICLE_MIN_CONFIDENCE = 0.35


# ============================================================
# BOUNDING BOX SMOOTHING
# ============================================================

SMOOTHING_ALPHA = 0.70

STABILITY_DEADBAND = 0.50

MAX_BOX_STEP = 150.00


# Position change = light smoothing
POSITION_ALPHA = 0.75

# Width/height change = even faster
SIZE_ALPHA = 0.85


# ============================================================
# VEHICLE CROP
# ============================================================

VEHICLE_CROP_PADDING = 10

BEST_CROP_MIN_AREA = 1000


# ============================================================
# DIRECTORIES
# ============================================================

ANPR_INPUT_DIR = "anpr_input"

EVENT_LOG_DIR = "events"


# ============================================================
# SSL
# ============================================================

CERT_FILE = "server_cert.pem"

KEY_FILE = "server_key.pem"


# ============================================================
# REMOTE CAMERAS
# ============================================================

MAX_PHONE_CAMERAS = 5


# ============================================================
# MJPEG
# ============================================================

MJPEG_JPEG_QUALITY = 75

MJPEG_INTERVAL = 0.06


# ============================================================
# WEBRTC / TURN
# ============================================================

STUN_URL = "stun:stun.relay.metered.ca:80"

TURN_URLS = [

    "turn:global.relay.metered.ca:80",

    "turn:global.relay.metered.ca:80?transport=tcp",

    "turn:global.relay.metered.ca:443",

    "turns:global.relay.metered.ca:443?transport=tcp",

]


def get_turn_username():

    return os.getenv(
        "TURN_USERNAME",
        ""
    ).strip()


def get_turn_credential():

    return os.getenv(
        "TURN_CREDENTIAL",
        ""
    ).strip()


def turn_configured():

    return bool(
        get_turn_username()
        and
        get_turn_credential()
    )


def create_rtc_configuration():

    ice_servers = [

        RTCIceServer(
            urls=STUN_URL
        )

    ]

    username = get_turn_username()

    credential = get_turn_credential()

    if username and credential:

        for turn_url in TURN_URLS:

            ice_servers.append(

                RTCIceServer(

                    urls=turn_url,

                    username=username,

                    credential=credential

                )

            )

    return RTCConfiguration(

        iceServers=ice_servers

    )


def get_browser_ice_servers():

    servers = [

        {
            "urls": STUN_URL
        }

    ]

    username = get_turn_username()

    credential = get_turn_credential()

    if username and credential:

        for turn_url in TURN_URLS:

            servers.append(

                {

                    "urls": turn_url,

                    "username": username,

                    "credential": credential

                }

            )

    return servers


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

shared_model = None

model_lock = threading.Lock()


# ============================================================
# CLOUDFLARE GLOBAL DATA
# ============================================================

cloudflared_process = None

cloudflared_reader_thread = None

public_tunnel_url = ""

cloudflare_browser_opened = False

cloudflared_lock = threading.Lock()


# ============================================================
# CLOUDFLARE QUICK TUNNEL FUNCTIONS
# ============================================================

def find_cloudflared():

    # --------------------------------------------------------
    # First preference:
    # C:\cloudflared\cloudflared.exe
    # --------------------------------------------------------

    if os.path.exists(CLOUDFLARED_PATH):

        return CLOUDFLARED_PATH


    # --------------------------------------------------------
    # Second preference:
    # cloudflared.exe in current project folder
    # --------------------------------------------------------

    local_exe = os.path.join(

        os.getcwd(),

        "cloudflared.exe"

    )


    if os.path.exists(local_exe):

        return local_exe


    # --------------------------------------------------------
    # Third preference:
    # cloudflared available in PATH
    # --------------------------------------------------------

    path_exe = shutil.which(

        "cloudflared"

    )


    if path_exe:

        return path_exe


    return None


def cloudflared_output_reader():

    global public_tunnel_url
    global cloudflare_browser_opened

    process = cloudflared_process


    if process is None:

        return


    try:

        for line in iter(

            process.stdout.readline,

            ""

        ):

            if not line:

                break


            line = line.rstrip()


            if line:

                print(

                    f"[CLOUDFLARE] {line}"

                )


            # ------------------------------------------------
            # Detect Quick Tunnel URL.
            # Example:
            # https://something.trycloudflare.com
            # ------------------------------------------------

            match = re.search(

                r"https://[a-zA-Z0-9-]+\.trycloudflare\.com",

                line

            )


            if match:

                url = match.group(0).rstrip("/")


                with cloudflared_lock:

                    public_tunnel_url = url


                print(

                    "\n"

                    "============================================================\n"

                    "          CLOUDFLARE QUICK TUNNEL READY\n"

                    "============================================================\n"

                )


                print(

                    "\nPUBLIC HEAD DASHBOARD:"

                )


                print(

                    public_tunnel_url + "/"

                )


                print(

                    "\nPUBLIC REMOTE CAMERA PAGE:"

                )


                print(

                    public_tunnel_url + "/phone"

                )


                print(

                    "\n============================================================\n"

                )


                # ------------------------------------------------
                # Automatically open public dashboard once.
                # ------------------------------------------------

                if not cloudflare_browser_opened:

                    cloudflare_browser_opened = True


                    def open_browser():

                        time.sleep(1.5)


                        try:

                            webbrowser.open(

                                public_tunnel_url + "/"

                            )

                            print(

                                "[CLOUDFLARE] "

                                "Public dashboard opened in browser."

                            )

                        except Exception as e:

                            print(

                                "[CLOUDFLARE] "

                                f"Browser open error: {e}"

                            )


                    threading.Thread(

                        target=open_browser,

                        daemon=True

                    ).start()


    except Exception as e:

        print(

            "[CLOUDFLARE] "

            f"Reader error: {e}"

        )


def start_cloudflare_tunnel():

    global cloudflared_process
    global cloudflared_reader_thread
    global public_tunnel_url
    global cloudflare_browser_opened


    if not AUTO_START_CLOUDFLARE:

        print(

            "[CLOUDFLARE] "

            "Automatic Cloudflare startup disabled."

        )

        return False


    with cloudflared_lock:

        # ----------------------------------------------------
        # Prevent duplicate Cloudflare processes.
        # ----------------------------------------------------

        if (

            cloudflared_process is not None

            and

            cloudflared_process.poll() is None

        ):

            print(

                "[CLOUDFLARE] "

                "Tunnel is already running."

            )

            return True


        public_tunnel_url = ""

        cloudflare_browser_opened = False


    exe = find_cloudflared()


    if exe is None:

        print(

            "\n[CLOUDFLARE] WARNING:\n"

            "cloudflared.exe was not found.\n"

            f"Expected location:\n{CLOUDFLARED_PATH}\n"

            "Server will continue using local HTTPS.\n"

        )

        return False


    print(

        "\n[CLOUDFLARE] "

        "Starting Quick Tunnel..."

    )


    print(

        f"[CLOUDFLARE] "

        f"Executable: {exe}"

    )


    command = [

        exe,

        "tunnel",

        "--url",

        f"https://127.0.0.1:{SERVER_PORT}",

        "--no-tls-verify"

    ]


    try:

        with cloudflared_lock:

            cloudflared_process = subprocess.Popen(

                command,

                stdout=subprocess.PIPE,

                stderr=subprocess.STDOUT,

                stdin=subprocess.DEVNULL,

                text=True,

                bufsize=1

            )


        cloudflared_reader_thread = threading.Thread(

            target=cloudflared_output_reader,

            daemon=True

        )


        cloudflared_reader_thread.start()


        print(

            "[CLOUDFLARE] "

            "Quick Tunnel process started."

        )


        print(

            "[CLOUDFLARE] "

            "Waiting for public URL..."

        )


        return True


    except Exception as e:

        print(

            "\n[CLOUDFLARE] "

            f"Could not start cloudflared: {e}\n"

        )


        with cloudflared_lock:

            cloudflared_process = None


        return False


def stop_cloudflare_tunnel():

    global cloudflared_process
    global cloudflared_reader_thread
    global public_tunnel_url


    with cloudflared_lock:

        process = cloudflared_process


        if process is None:

            return


        print(

            "\n[CLOUDFLARE] "

            "Stopping Quick Tunnel..."

        )


        try:

            if process.poll() is None:

                process.terminate()


                try:

                    process.wait(

                        timeout=5

                    )

                except subprocess.TimeoutExpired:

                    print(

                        "[CLOUDFLARE] "

                        "Process did not stop normally. "

                        "Killing it..."

                    )


                    process.kill()


                    try:

                        process.wait(

                            timeout=2

                        )

                    except Exception:

                        pass


        except Exception as e:

            print(

                "[CLOUDFLARE] "

                f"Shutdown error: {e}"

            )


        cloudflared_process = None

        cloudflared_reader_thread = None

        public_tunnel_url = ""


    print(

        "[CLOUDFLARE] "

        "Quick Tunnel stopped."

    )


# ============================================================
# SHARED YOLO MODEL
# ============================================================

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

        x509.DNSName(
            "localhost"
        ),

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

        .subject_name(
            subject
        )

        .issuer_name(
            issuer
        )

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

            +

            timedelta(
                days=365
            )

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

    track_confirmations: dict = field(

        default_factory=dict

    )

    track_last_seen: dict = field(

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
    # Dimensions
    # --------------------------------------------------------

    frame_width: int = 0

    frame_height: int = 0


    # --------------------------------------------------------
    # WebRTC
    # --------------------------------------------------------

    connection_state: str = "new"

    ice_connection_state: str = "new"

    ice_gathering_state: str = "new"

    video_received: bool = False

    last_frame_time: float = 0.0

    track_error: str = ""

    cleanup_started: bool = False


    def initialize_model(self):

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


    old = session.best_vehicle_crops.get(

        track_id

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


    previous = session.smooth_boxes.get(

        track_id

    )


    if previous is None:

        session.smooth_boxes[

            track_id

        ] = current

        return current.astype(int)


    diff = current - previous


    # Small movement = don't move box.
    if np.all(

        np.abs(diff)

        <= STABILITY_DEADBAND

    ):

        return previous.astype(int)


    distance = np.linalg.norm(
        diff
    )


    # Limit sudden jumps.
    if distance > MAX_BOX_STEP:

        ratio = (

            MAX_BOX_STEP

            /

            distance

        )


        current = (

            previous

            +

            diff * ratio

        )


    smoothed = (

        SMOOTHING_ALPHA

        *

        current

        +

        (

            1

            -

            SMOOTHING_ALPHA

        )

        *

        previous

    )


    session.smooth_boxes[

        track_id

    ] = smoothed


    return smoothed.astype(int)


# ============================================================
# BOX HELPERS
# ============================================================

def box_area(box):

    x1, y1, x2, y2 = box

    width = max(
        0,
        x2 - x1
    )

    height = max(
        0,
        y2 - y1
    )

    return width * height


def intersection_area(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a

    bx1, by1, bx2, by2 = box_b


    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )


    if (

        ix2 <= ix1

        or

        iy2 <= iy1

    ):

        return 0.0


    return (

        ix2 - ix1

    ) * (

        iy2 - iy1

    )


def is_box_inside(

    small_box,

    large_box,

    threshold=0.60

):

    small_area = box_area(
        small_box
    )


    if small_area <= 0:

        return False


    overlap = intersection_area(

        small_box,

        large_box

    )


    return (

        overlap / small_area

    ) >= threshold


# ============================================================
# PERSON DETECTION FILTER
# ============================================================

def filter_person_detections(

    person_candidates,

    frame_width,

    frame_height

):

    """
    Removes obvious false person detections.

    Input:
        [
            {
                "track_id": ...,
                "box": (...),
                "confidence": ...
            }
        ]

    Output:
        filtered list
    """


    filtered = []


    # --------------------------------------------------------
    # STEP 1 - Size / confidence filter
    # --------------------------------------------------------

    for candidate in person_candidates:

        x1, y1, x2, y2 = candidate["box"]

        width = max(
            0,
            x2 - x1
        )

        height = max(
            0,
            y2 - y1
        )

        area = width * height


        if (

            candidate["confidence"]

            <

            PERSON_MIN_CONFIDENCE

        ):

            continue


        # Dynamic minimum for very small videos.
        dynamic_min_height = max(

            45,

            int(frame_height * 0.045)

        )


        minimum_height = max(

            MIN_PERSON_HEIGHT,

            dynamic_min_height

        )


        if height < minimum_height:

            continue


        if area < MIN_PERSON_AREA:

            continue


        # A person box normally should not be
        # extremely wide compared with height.
        aspect_ratio = (

            width / max(
                height,
                1
            )

        )


        if aspect_ratio > 2.8:

            continue


        filtered.append(
            candidate
        )


    # --------------------------------------------------------
    # STEP 2 - Remove nested false detections
    # --------------------------------------------------------

    final_candidates = []


    for i, candidate in enumerate(
        filtered
    ):

        candidate_box = candidate[
            "box"
        ]

        candidate_area = box_area(
            candidate_box
        )


        reject = False


        for j, other in enumerate(
            filtered
        ):

            if i == j:

                continue


            other_box = other[
                "box"
            ]

            other_area = box_area(
                other_box
            )


            # Only compare when other box is significantly larger.
            if other_area <= (

                candidate_area * 1.25

            ):

                continue


            # If the small person box is mostly
            # inside the larger person box,
            # it is almost certainly a false
            # duplicate / hand / object detection.
            if is_box_inside(

                candidate_box,

                other_box,

                PERSON_CONTAINMENT_THRESHOLD

            ):

                reject = True

                break


        if not reject:

            final_candidates.append(
                candidate
            )


    return final_candidates


# ============================================================
# PERSON TRACK CONFIRMATION
# ============================================================

def confirm_person_track(

    session,

    track_id

):

    now = time.time()


    last_seen = session.track_last_seen.get(

        track_id

    )


    # If this ID disappeared for too long,
    # start confirmation again.
    if (

        last_seen is None

        or

        now - last_seen > 1.0

    ):

        session.track_confirmations[
            track_id
        ] = 1

    else:

        session.track_confirmations[
            track_id
        ] = (

            session.track_confirmations.get(
                track_id,
                0
            )

            +

            1

        )


    session.track_last_seen[
        track_id
    ] = now


    return (

        session.track_confirmations.get(
            track_id,
            0
        )

        >=

        PERSON_CONFIRM_FRAMES

    )


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


        with model_lock:

            results = model.track(

                source=frame,

                persist=True,

                tracker="bytetrack.yaml",

                conf=CONFIDENCE,

                imgsz=IMAGE_SIZE,

                classes=list(
                    ALLOWED_CLASSES
                ),

                iou=0.50,

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


    person_candidates = []


    vehicle_candidates = []


    # ========================================================
    # READ YOLO RESULTS
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


                raw_box = (

                    boxes.xyxy[i]

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

                            boxes.id[i].item()

                        )

                    except Exception:

                        track_id = None


                if track_id is None:

                    # Temporary fallback ID.
                    track_id = 100000 + i


                current_ids.add(
                    track_id
                )


                candidate = {

                    "track_id":
                        track_id,

                    "box": (

                        x1,

                        y1,

                        x2,

                        y2

                    ),

                    "confidence":
                        confidence,

                    "class_id":
                        cls_id

                }


                if cls_id == PERSON:

                    if confidence >= PERSON_MIN_CONFIDENCE:

                        person_candidates.append(
                            candidate
                        )


                elif cls_id in VEHICLE_CLASSES:

                    if confidence >= VEHICLE_MIN_CONFIDENCE:

                        vehicle_candidates.append(
                            candidate
                        )


            except Exception as e:

                print(

                    f"[{session.camera_id}] "

                    f"Detection parsing error: {e}"

                )


    # ========================================================
    # FILTER PERSON DETECTIONS
    # ========================================================

    person_candidates = filter_person_detections(

        person_candidates,

        w,

        h

    )


    # ========================================================
    # DRAW PERSONS
    # ========================================================

    for candidate in person_candidates:

        track_id = candidate[
            "track_id"
        ]

        x1, y1, x2, y2 = candidate[
            "box"
        ]

        confidence = candidate[
            "confidence"
        ]


        # Confirm detection across multiple frames.
        confirmed = confirm_person_track(

            session,

            track_id

        )


        if not confirmed:

            # Do not display unstable first-frame
            # detections.
            continue


        person_count += 1


        # Smooth box.
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


        cx = int(

            (sx1 + sx2) / 2

        )


        cy = int(

            (sy1 + sy2) / 2

        )


        # RED person box.
        box_color = (

            0,

            0,

            255

        )


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


        label = (

            f"PERSON "

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


        session.previous_boxes[
            track_id
        ] = (

            sx1,

            sy1,

            sx2,

            sy2

        )


    # ========================================================
    # DRAW VEHICLES
    # ========================================================

    for candidate in vehicle_candidates:

        track_id = candidate[
            "track_id"
        ]

        x1, y1, x2, y2 = candidate[
            "box"
        ]

        confidence = candidate[
            "confidence"
        ]

        cls_id = candidate[
            "class_id"
        ]


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


        cx = int(

            (sx1 + sx2) / 2

        )


        cy = int(

            (sy1 + sy2) / 2

        )


        vehicle_count += 1


        class_name = CLASS_NAMES.get(

            cls_id,

            "VEHICLE"

        )


        # Save best vehicle crop.
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

            class_name,

            confidence

        )


        # BLUE vehicle box.
        box_color = (

            255,

            0,

            0

        )


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


        label = (

            f"{class_name} "

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


        session.previous_boxes[
            track_id
        ] = (

            sx1,

            sy1,

            sx2,

            sy2

        )


    # ========================================================
    # CLEAN OLD TRACK DATA
    # ========================================================

    now = time.time()


    old_ids = list(

        session.previous_boxes.keys()

    )


    for track_id in old_ids:

        last_seen = session.track_last_seen.get(

            track_id,

            now

        )


        if (

            track_id not in current_ids

            and

            now - last_seen > 1.0

        ):

            session.previous_boxes.pop(

                track_id,

                None

            )


            session.smooth_boxes.pop(

                track_id,

                None

            )


            session.track_confirmations.pop(

                track_id,

                None

            )


            session.track_last_seen.pop(

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

        (20, 115),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        (255, 255, 255),

        2

    )


    cv2.putText(

        output,

        f"VEHICLES: {vehicle_count}",

        (20, 145),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        (255, 255, 255),

        2

    )


    cv2.putText(

        output,

        f"EVENTS  : {session.event_count}",

        (20, 175),

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


        with session.frame_lock:

            if session.latest_frame is not None:

                frame = (

                    session.latest_frame.copy()

                )

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
# PHONE / REMOTE WEBRTC TRACK
# ============================================================

async def read_phone_track(

    session,

    track

):

    print(

        f"[{session.camera_id}] "

        f"Video track started."

    )


    try:

        while session.active:

            frame = await track.recv()


            image = frame.to_ndarray(

                format="bgr24"

            )


            session.video_received = True

            session.last_frame_time = time.time()

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


            with session.frame_lock:

                session.latest_frame = image


    except asyncio.CancelledError:

        pass


    except Exception as e:

        session.track_error = str(e)


        print(

            f"[{session.camera_id}] "

            f"Video track ended: {e}"

        )


    finally:

        print(

            f"[{session.camera_id}] "

            f"Video track reader stopped."

        )


# ============================================================
# LAPTOP CAMERA CAPTURE
# ============================================================

def laptop_capture_loop(session):

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


        session.video_received = True

        session.last_frame_time = time.time()


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


        session.capture_stop = threading.Event()


        session.initialize_model()


        session.worker_task = asyncio.create_task(

            camera_worker(

                session

            )

        )


        session.capture_thread = threading.Thread(

            target=laptop_capture_loop,

            args=(session,),

            daemon=True

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

.feed {
    width: 100%;
    height: auto;
    display: block;
    background: black;
    object-fit: contain;
    max-width: 100%;
    aspect-ratio: auto;
}

.info {
    padding: 10px 15px;
    color: #aab5c1;
    font-size: 13px;
    line-height: 1.6;
}

.live {
    color: #48d597;
    font-weight: bold;
}

.connecting {
    color: #ffd166;
    font-weight: bold;
}

.error {
    color: #ff6b6b;
    font-weight: bold;
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
OPEN CAMERA PAGE
</button>

</div>

<div class="status" id="status">
Loading camera status...
</div>

<div class="url-box">

<strong>
REMOTE CAMERA URL:
</strong>

<span id="phoneUrl">
Loading...
</span>

<br><br>

Open this URL on a phone or another laptop to use its camera.

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

function escapeHtml(value) {

    const div =
        document.createElement("div");

    div.textContent =
        value ?? "";

    return div.innerHTML;
}


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


function getStateHtml(camera) {

    if (
        camera.source_type === "LAPTOP"
    ) {

        return `
            <span class="live">
                LOCAL CAMERA
            </span>
        `;

    }


    if (
        camera.video_received &&
        camera.connection_state === "connected"
    ) {

        return `
            <span class="live">
                ● LIVE
            </span>
        `;

    }


    if (
        camera.connection_state === "failed"
        ||
        camera.connection_state === "closed"
    ) {

        return `
            <span class="error">
                ● ${escapeHtml(camera.connection_state)}
            </span>
        `;

    }


    return `
        <span class="connecting">
            ● CONNECTING...
        </span>
    `;

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
            "HEAD | Connected cameras: "
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


                const state =
                    getStateHtml(camera);


                card.innerHTML =

                    '<div class="camera-title">'
                    +
                    escapeHtml(
                        camera.camera_id
                    )
                    +
                    ' — '
                    +
                    '<span>'
                    +
                    escapeHtml(
                        camera.source_type
                    )
                    +
                    '</span>'
                    +
                    ' &nbsp; '
                    +
                    state
                    +
                    '</div>'

                    +

                    '<img '
                    +
                    'class="feed" '
                    +
                    'src="/mjpeg/'
                    +
                    encodeURIComponent(
                        camera.camera_id
                    )
                    +
                    '" '
                    +
                    'alt="Camera Feed">'

                    +

                    '<div class="info">'
                    +

                    'Connection: '
                    +
                    escapeHtml(
                        camera.connection_state
                    )

                    +

                    '<br>ICE: '
                    +
                    escapeHtml(
                        camera.ice_connection_state
                    )

                    +

                    '<br>Video Received: '
                    +
                    (
                        camera.video_received
                        ? "YES"
                        : "NO"
                    )

                    +

                    '<br>FPS: '
                    +
                    Number(
                        camera.fps || 0
                    ).toFixed(1)

                    +

                    '<br>Events: '
                    +
                    camera.events

                    +

                    '<br>Processed Frames: '
                    +
                    camera.processed_frames

                    +

                    '<br>Resolution: '
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
# PHONE / REMOTE CAMERA HTML
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

<title>IBVAP Remote Camera</title>

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

.stop {
    background: #d92d20;
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

.debug {

    width: 95%;

    max-width: 900px;

    margin: 10px auto;

    text-align: left;

    color: #7f8c9a;

    font-size: 12px;

    background: #10161d;

    padding: 10px;

    border-radius: 8px;
}

</style>

</head>

<body>

<h1>
IBVAP Remote Camera
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

<div class="debug">

<div>
ICE: <span id="iceState">-</span>
</div>

<div>
Connection: <span id="connectionState">-</span>
</div>

<div>
Signaling: <span id="signalingState">-</span>
</div>

</div>

<br>

<button onclick="startCamera()">
START CAMERA
</button>

<button
class="stop"
onclick="stopCamera()"
>
STOP CAMERA
</button>


<script>

let localStream = null;

let pc = null;

let cameraId = null;


function setStatus(text) {

    document.getElementById(
        "status"
    ).innerText = text;

}


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


async function getWebRTCConfig() {

    const response =
        await fetch(
            "/api/webrtc-config"
        );


    if (!response.ok) {

        throw new Error(

            "Could not load WebRTC configuration."

        );

    }


    return await response.json();

}


async function startCamera() {

    try {

        stopCamera(false);


        setStatus(
            "Requesting camera permission..."
        );


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


        const videoTrack =
            localStream.getVideoTracks()[0];


        if (videoTrack) {

            const settings =
                videoTrack.getSettings();


            console.log(

                "Camera settings:",

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


        setStatus(
            "Loading WebRTC..."
        );


        const rtcConfig =
            await getWebRTCConfig();


        console.log(
            "WebRTC configuration loaded."
        );


        pc =
            new RTCPeerConnection(
                rtcConfig
            );


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

                const state =
                    pc.connectionState;


                console.log(

                    "Connection state:",

                    state

                );


                document.getElementById(
                    "connectionState"
                ).innerText =
                    state;


                if (
                    state === "connected"
                ) {

                    setStatus(
                        "CONNECTED"
                    );

                }

                else if (
                    state === "connecting"
                ) {

                    setStatus(
                        "CONNECTING..."
                    );

                }

                else if (
                    state === "disconnected"
                ) {

                    setStatus(
                        "DISCONNECTED"
                    );

                }

                else if (
                    state === "failed"
                ) {

                    setStatus(
                        "CONNECTION FAILED"
                    );

                }

                else if (
                    state === "closed"
                ) {

                    setStatus(
                        "CLOSED"
                    );

                }

            };


        pc.oniceconnectionstatechange =
            () => {

                const state =
                    pc.iceConnectionState;


                console.log(

                    "ICE state:",

                    state

                );


                document.getElementById(
                    "iceState"
                ).innerText =
                    state;

            };


        pc.onsignalingstatechange =
            () => {

                document.getElementById(
                    "signalingState"
                ).innerText =
                    pc.signalingState;

            };


        pc.onicecandidateerror =
            event => {

                console.warn(

                    "ICE candidate error:",

                    event

                );

            };


        setStatus(
            "Creating secure connection..."
        );


        const offer =
            await pc.createOffer();


        await pc.setLocalDescription(
            offer
        );


        setStatus(
            "Finding best network path..."
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

                    body:
                        JSON.stringify({

                            sdp:
                                pc.localDescription.sdp,

                            type:
                                pc.localDescription.type

                        })

                }
            );


        if (!response.ok) {

            let errorText =
                "Server rejected connection.";


            try {

                const errorData =
                    await response.json();


                if (
                    errorData.error
                ) {

                    errorText =
                        errorData.error;

                }

            }

            catch (_) {}


            throw new Error(
                errorText
            );

        }


        const answer =
            await response.json();


        await pc.setRemoteDescription(
            answer
        );


        cameraId =
            answer.camera_id;


        document.getElementById(
            "cameraId"
        ).innerText =
            cameraId;


        setStatus(
            "SIGNALING OK — CONNECTING..."
        );


        console.log(

            "Camera ID:",

            cameraId

        );

    }


    catch (error) {

        console.error(
            error
        );


        setStatus(

            "ERROR: "
            +
            error.message

        );


        alert(

            "Camera connection failed:\n"
            +
            error.message

        );

    }

}


function stopCamera(
    showStatus = true
) {

    if (localStream) {

        localStream
            .getTracks()
            .forEach(

                track => track.stop()

            );


        localStream = null;

    }


    if (pc) {

        try {

            pc.close();

        }

        catch (_) {}


        pc = null;

    }


    document.getElementById(
        "video"
    ).srcObject =
        null;


    if (showStatus) {

        setStatus(
            "STOPPED"
        );

    }


    document.getElementById(
        "cameraId"
    ).innerText =
        "-";


    document.getElementById(
        "resolution"
    ).innerText =
        "Resolution: -";


    document.getElementById(
        "iceState"
    ).innerText =
        "-";


    document.getElementById(
        "connectionState"
    ).innerText =
        "-";


    document.getElementById(
        "signalingState"
    ).innerText =
        "-";

}

</script>

</body>

</html>
"""


# ============================================================
# DASHBOARD ROUTE
# ============================================================

async def dashboard_handler(request):

    return web.Response(

        text=DASHBOARD_HTML,

        content_type="text/html"

    )


# ============================================================
# PHONE ROUTE
# ============================================================

async def phone_handler(request):

    return web.Response(

        text=PHONE_HTML,

        content_type="text/html"

    )


# ============================================================
# API INFO
# ============================================================

async def info_handler(request):

    local_ip = get_local_ip()


    # --------------------------------------------------------
    # Use Cloudflare public URL when available.
    # Otherwise use local IP.
    # --------------------------------------------------------

    with cloudflared_lock:

        tunnel_url = public_tunnel_url


    if tunnel_url:

        base_url = tunnel_url.rstrip("/")

    else:

        base_url = (

            f"https://"

            f"{local_ip}:"

            f"{SERVER_PORT}"

        )


    return web.json_response({

        "name":
            "IBVAP",

        "phone_url":
            base_url + "/phone",

        "dashboard_url":
            base_url + "/",

        "local_ip":
            local_ip,

        "port":
            SERVER_PORT,

        "turn_configured":
            turn_configured(),

        "public_tunnel_url":
            tunnel_url

    })


# ============================================================
# API WEBRTC CONFIG
# ============================================================

async def webrtc_config_handler(request):

    return web.json_response({

        "iceServers":
            get_browser_ice_servers()

    })


# ============================================================
# API CAMERAS
# ============================================================

async def cameras_handler(request):

    result = []


    with sessions_lock:

        current_sessions = list(

            sessions.values()

        )


    for session in current_sessions:

        last_frame_age = None


        if session.last_frame_time > 0:

            last_frame_age = (

                time.time()

                -

                session.last_frame_time

            )


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
                session.frame_height,

            "connection_state":
                session.connection_state,

            "ice_connection_state":
                session.ice_connection_state,

            "ice_gathering_state":
                session.ice_gathering_state,

            "video_received":
                session.video_received,

            "last_frame_age":
                last_frame_age,

            "track_error":
                session.track_error

        })


    return web.json_response({

        "cameras":
            result

    })


# ============================================================
# LAPTOP START API
# ============================================================

async def laptop_start_handler(request):

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

async def laptop_stop_handler(request):

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

async def offer_handler(request):

    # --------------------------------------------------------
    # READ OFFER
    # --------------------------------------------------------

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
    # CAMERA LIMIT
    # --------------------------------------------------------

    with sessions_lock:

        phone_count = len([

            s

            for s in sessions.values()

            if s.source_type == "REMOTE"

        ])


    if phone_count >= MAX_PHONE_CAMERAS:

        return web.json_response(

            {

                "error":
                    "Maximum remote camera limit reached."

            },

            status=503

        )


    # --------------------------------------------------------
    # NEW CAMERA
    # --------------------------------------------------------

    camera_id = (

        get_new_phone_camera_id()

    )


    session = CameraSession(

        camera_id=camera_id,

        source_type="REMOTE"

    )


    # --------------------------------------------------------
    # WEBRTC PEER
    # --------------------------------------------------------

    pc = RTCPeerConnection(

        configuration=
            create_rtc_configuration()

    )


    session.pc = pc


    session.connection_state = (

        pc.connectionState

    )


    session.ice_connection_state = (

        pc.iceConnectionState

    )


    session.ice_gathering_state = (

        pc.iceGatheringState

    )


    with sessions_lock:

        sessions[

            camera_id

        ] = session


    print(

        f"[{camera_id}] "

        f"New remote camera connection."

    )


    # --------------------------------------------------------
    # TRACK
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
    # ICE
    # --------------------------------------------------------

    @pc.on("iceconnectionstatechange")

    async def on_iceconnectionstatechange():

        session.ice_connection_state = (

            pc.iceConnectionState

        )


        print(

            f"[{camera_id}] "

            f"ICE state: "

            f"{pc.iceConnectionState}"

        )


    # --------------------------------------------------------
    # ICE GATHERING
    # --------------------------------------------------------

    @pc.on("icegatheringstatechange")

    async def on_icegatheringstatechange():

        session.ice_gathering_state = (

            pc.iceGatheringState

        )


    # --------------------------------------------------------
    # CONNECTION STATE
    # --------------------------------------------------------

    @pc.on("connectionstatechange")

    async def on_connectionstatechange():

        state = pc.connectionState


        session.connection_state = state


        print(

            f"[{camera_id}] "

            f"Connection state: "

            f"{state}"

        )


        if state in (

            "failed",

            "closed"

        ):

            await cleanup_phone_session(

                camera_id

            )


    # --------------------------------------------------------
    # SDP
    # --------------------------------------------------------

    try:

        await pc.setRemoteDescription(

            offer

        )


        answer = await pc.createAnswer()


        await pc.setLocalDescription(

            answer

        )


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

            f"WebRTC error: "

            f"{e}"

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
# CLEANUP REMOTE CAMERA
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


    if session.cleanup_started:

        return


    session.cleanup_started = True


    print(

        f"[{camera_id}] "

        f"Cleaning up..."

    )


    session.active = False


    if session.pc:

        try:

            if (

                session.pc.connectionState

                !=

                "closed"

            ):

                await session.pc.close()

        except Exception:

            pass


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

            try:

                session.worker_task.cancel()

            except Exception:

                pass


    save_session_events(

        session

    )


    save_all_best_vehicle_crops(

        session

    )


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

async def mjpeg_handler(request):

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

                        b"Content-Type: image/jpeg\r\n"

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

async def events_handler(request):

    return web.json_response({

        "events":
            global_event_logs

    })


# ============================================================
# SHUTDOWN
# ============================================================

async def shutdown_server(app):

    print(

        "\n[SERVER] "

        "Shutting down..."

    )


    shutdown_event.set()


    try:

        await stop_laptop_camera()

    except Exception:

        pass


    with sessions_lock:

        phone_ids = [

            camera_id

            for camera_id, session

            in sessions.items()

            if session.source_type == "REMOTE"

        ]


    for camera_id in phone_ids:

        try:

            await cleanup_phone_session(

                camera_id

            )

        except Exception:

            pass


    save_all_events()


    # --------------------------------------------------------
    # STOP CLOUDFLARE QUICK TUNNEL
    # --------------------------------------------------------

    try:

        stop_cloudflare_tunnel()

    except Exception as e:

        print(

            "[CLOUDFLARE] "

            f"Shutdown error: {e}"

        )


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

        "/api/webrtc-config",

        webrtc_config_handler

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


    # --------------------------------------------------------
    # START CLOUDFLARE WHEN AIOHTTP APP STARTS
    # --------------------------------------------------------

    async def on_startup(app):

        if AUTO_START_CLOUDFLARE:

            start_cloudflare_tunnel()


    app.on_startup.append(

        on_startup

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


    # --------------------------------------------------------
    # DIRECTORIES
    # --------------------------------------------------------

    ensure_directories()


    # --------------------------------------------------------
    # MODEL CHECK
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
    # LOAD YOLO
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
    # SSL CONTEXT
    # --------------------------------------------------------

    ssl_context = ssl.create_default_context(

        ssl.Purpose.CLIENT_AUTH

    )


    ssl_context.load_cert_chain(

        CERT_FILE,

        KEY_FILE

    )


    # --------------------------------------------------------
    # URLS
    # --------------------------------------------------------

    local_ip = get_local_ip()


    dashboard_url = (

        f"https://"

        f"{local_ip}:"

        f"{SERVER_PORT}/"

    )


    phone_url = (

        f"https://"

        f"{local_ip}:"

        f"{SERVER_PORT}/phone"

    )


    # --------------------------------------------------------
    # STARTUP INFORMATION
    # --------------------------------------------------------

    print(

        "\n============================================================"

    )


    print(

        "\nHEAD DASHBOARD:"

    )


    print(

        dashboard_url

    )


    print(

        "\nREMOTE CAMERA PAGE:"

    )


    print(

        phone_url

    )


    print(

        "\nCLOUDFLARE:"

    )


    if AUTO_START_CLOUDFLARE:

        print(

            "AUTO START ENABLED"

        )

        print(

            f"Executable: {CLOUDFLARED_PATH}"

        )

        print(

            "Public URL will appear automatically."

        )

    else:

        print(

            "AUTO START DISABLED"

        )


    print(

        "\nTURN STATUS:"

    )


    if turn_configured():

        print(

            "TURN + STUN CONFIGURED"

        )

    else:

        print(

            "WARNING: TURN CREDENTIALS NOT CONFIGURED"

        )


        print(

            "Remote internet cameras may NOT work."

        )


    print(

        "\n============================================================"

    )


    print(

        "\nIMPORTANT:"

    )


    print(

        "1. This laptop is the HEAD for this session."

    )


    print(

        "2. Other members do NOT need to run main.py."

    )


    print(

        "3. They open the REMOTE CAMERA PAGE in browser."

    )


    print(

        "4. Phone or laptop webcam can be used."

    )


    print(

        "5. Multiple remote cameras are supported."

    )


    print(

        "6. TURN + WebRTC is enabled when credentials exist."

    )


    print(

        "7. Laptop webcam can be started from dashboard."

    )


    print(

        "8. Camera aspect ratio is preserved."

    )


    print(

        "9. Person detection uses filtering + confirmation + smoothing."

    )


    print(

        "10. Cloudflare Quick Tunnel starts automatically."

    )


    print(

        "11. Public dashboard opens automatically when tunnel is ready."

    )


    print(

        "\n============================================================\n"

    )


    # --------------------------------------------------------
    # APP
    # --------------------------------------------------------

    app = create_app()


    # --------------------------------------------------------
    # SERVER
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