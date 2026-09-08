import os
import cv2
import asyncio
import json
from aiohttp import web
import aiortc
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from av import VideoFrame
from ultralytics import YOLO

# Face engine module integration
from face_engine import identify_face

# --- INITIALIZATION ---
# YOLO Model initialize
model = YOLO('yolov8n.pt')

# SSL Certificate Check & Paths
CERT_FILE = 'server_cert.pem'
KEY_FILE = 'server_key.pem'

# Set of active WebRTC connections
pcs = set()

# --- FRAME PROCESSING FUNCTION ---
def process_frame(frame, prev_boxes_dict):
    annotated_frame = frame.copy()
    
    # YOLO Tracking call (conf=0.60 threshold se noise/fingers eliminate hote hain)
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=0.60)

    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            conf = float(box.conf[0].cpu().numpy())
            if conf < 0.60:
                continue

            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0].cpu().numpy())
            label = model.names[cls_id]

            x1, y1, x2, y2 = xyxy

            # --- PERSON FACE IDENTIFICATION ---
            if label == 'person':
                person_crop = frame[y1:y2, x1:x2]
                person_name = identify_face(person_crop)

                if person_name != "UNKNOWN":
                    display_text = f"KNOWN: {person_name}"
                    color = (0, 255, 0)  # Bright Green for Known
                else:
                    display_text = "UNKNOWN PERSON"
                    color = (0, 0, 255)  # Bright Red for Unknown

                # 1. Custom Bounding Box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)

                # 2. Text Label Solid Badge (Clear Visibility)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)

                badge_y1 = max(0, y1 - text_h - 12)
                cv2.rectangle(
                    annotated_frame, 
                    (x1, badge_y1), 
                    (x1 + text_w + 10, y1), 
                    color, 
                    -1
                )

                # 3. High-Contrast White Text Overlay
                cv2.putText(
                    annotated_frame, 
                    display_text, 
                    (x1 + 5, y1 - 6 if y1 - 6 > text_h else y1 + text_h + 5), 
                    font, 
                    font_scale, 
                    (255, 255, 255), 
                    thickness, 
                    cv2.LINE_AA
                )

    return annotated_frame


# --- WEBRTC VIDEO TRANSFORM TRACK ---
class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.prev_boxes = {}

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        # Frame Processing
        annotated = process_frame(img, self.prev_boxes)

        new_frame = VideoFrame.from_ndarray(annotated, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame


# --- ROUTES & SERVER HANDLERS ---
async def index(request):
    content = open(os.path.join(os.path.dirname(__file__), "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            pc.addTrack(VideoTransformTrack(track))

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })

async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    app = web.Application()
    app.on_shutdown.append(on_shutdown)

    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.router.add_static("/", path=os.path.dirname(__file__), name="static")

    import ssl
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(CERT_FILE, KEY_FILE)

    print("IBVAP Dashboard running at https://0.0.0.0:8443")
    web.run_app(app, host="0.0.0.0", port=8443, ssl_context=ssl_context)
