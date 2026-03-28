"""
Smart Gov Biometric API  ·  main.py
====================================
Single entry point. Run with:
    uvicorn main:app --reload --port 8000

All routes:
  GET  /                          Health check
  POST /persons                   Register a person
  GET  /persons/{national_id}     Look up a person
  GET  /persons/{national_id}/profiles  View enrolled face profiles
  POST /enroll-faces              Enrol reference gallery
  POST /verify-face               Full biometric verification
  WS   /ws/stream/{national_id}  Real-time frame analysis (browser webcam)
"""

import os
import io
import uuid
import shutil
import base64
import json

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from face_service import (
    smart_verify,
    enroll_reference_gallery,
    create_person,
    get_person_by_national_id,
    get_face_profiles,
)
from face_core import (
    get_face_embedding_from_array,
    estimate_head_pose,
    classify_pose,
    assess_texture_liveness,
    _quality_from_array,
)

# ─────────────────────────────────────────────
#  APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title="Smart Gov Biometric API",
    description="Face verification, enrolment, and person management.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def save_temp_file(upload_file: UploadFile) -> str:
    ext = os.path.splitext(upload_file.filename)[1] if upload_file.filename else ".jpg"
    path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}{ext}")
    with open(path, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    return path


def cleanup(paths: list):
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


def decode_frame(b64_data: str) -> Optional[np.ndarray]:
    """Decode a base64 JPEG/PNG frame from the browser."""
    try:
        if "," in b64_data:
            b64_data = b64_data.split(",")[1]
        img_bytes = base64.b64decode(b64_data)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ─────────────────────────────────────────────
#  HEALTH CHECK
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Confirm API is alive."""
    return {"status": "ok", "message": "Smart Gov Biometric API v2 running"}


# ─────────────────────────────────────────────
#  PERSON MANAGEMENT
# ─────────────────────────────────────────────

class PersonCreate(BaseModel):
    national_id: str
    first_name: str
    last_name: str
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None


@app.post("/persons", tags=["Persons"])
def register_person(body: PersonCreate):
    """
    Register a new person.
    Send JSON body with national_id, first_name, last_name, and optional fields.
    """
    try:
        person = create_person(**body.model_dump())
        return {"success": True, "person": person}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/persons/{national_id}", tags=["Persons"])
def lookup_person(national_id: str):
    """Look up a person by national ID. Returns 404 if not found."""
    person = get_person_by_national_id(national_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    return {"success": True, "person": person}


@app.get("/persons/{national_id}/profiles", tags=["Enrolment"])
def get_enrolled_profiles(national_id: str):
    """
    List enrolled face profiles (poses and quality scores) for a person.
    Embeddings are not returned for security.
    """
    person = get_person_by_national_id(national_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    profiles = get_face_profiles(person["id"])
    safe = [
        {
            "id": p["id"],
            "pose": p["pose"],
            "quality_score": p.get("quality_score"),
            "quality_issues": p.get("quality_issues"),
            "created_at": p.get("created_at"),
        }
        for p in profiles
    ]
    return {"success": True, "profiles": safe}


# ─────────────────────────────────────────────
#  FACE ENROLMENT
# ─────────────────────────────────────────────

@app.post("/enroll-faces", tags=["Enrolment"])
async def enroll_faces(
    national_id: str = Form(...),
    front: UploadFile = File(...),
    left: UploadFile = File(None),
    right: UploadFile = File(None),
):
    """
    Enrol reference face images for a person.

    Requires front image. Left and right are optional but improve accuracy.
    Person must already be registered via POST /persons.

    Form fields:
        national_id (str)
        front       (file, required)
        left        (file, optional)
        right       (file, optional)
    """
    saved = []
    try:
        person = get_person_by_national_id(national_id)
        if not person:
            raise HTTPException(
                status_code=404,
                detail="Person not found. Register them first via POST /persons.",
            )

        gallery = {}
        front_path = save_temp_file(front)
        saved.append(front_path)
        gallery["front"] = front_path

        if left and left.filename:
            p = save_temp_file(left)
            saved.append(p)
            gallery["left"] = p

        if right and right.filename:
            p = save_temp_file(right)
            saved.append(p)
            gallery["right"] = p

        result = enroll_reference_gallery(person["id"], gallery)
        return {"success": True, "enrolled": len(result), "poses": list(gallery.keys())}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup(saved)


# ─────────────────────────────────────────────
#  FACE VERIFICATION (file upload)
# ─────────────────────────────────────────────

@app.post("/verify-face", tags=["Verification"])
async def verify_face(
    national_id: str = Form(...),
    attempts: int = Form(1),
    known_device: bool = Form(True),
    blink_detected: bool = Form(False),
    front: UploadFile = File(...),
    left: UploadFile = File(None),
    right: UploadFile = File(None),
):
    """
    Verify a live face against the enrolled gallery.

    Form fields:
        national_id    (str)
        attempts       (int, default 1)
        known_device   (bool, default true)
        blink_detected (bool, default false — set true if browser confirmed blink)
        front          (file, required)
        left           (file, optional — improves liveness scoring)
        right          (file, optional — improves liveness scoring)

    Returns:
        score, decision, biometric breakdown, anti-spoofing report
    """
    saved = []
    try:
        person = get_person_by_national_id(national_id)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        image_paths = []
        front_path = save_temp_file(front)
        saved.append(front_path)
        image_paths.append(front_path)

        if left and left.filename:
            p = save_temp_file(left)
            saved.append(p)
            image_paths.append(p)

        if right and right.filename:
            p = save_temp_file(right)
            saved.append(p)
            image_paths.append(p)

        result = smart_verify(
            user_id=person["id"],
            live_images=image_paths,
            attempts=attempts,
            known_device=known_device,
            blink_detected=blink_detected,
        )
        return {"success": True, "result": result}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup(saved)


# ─────────────────────────────────────────────
#  WEBSOCKET: REAL-TIME FRAME ANALYSIS
# ─────────────────────────────────────────────

@app.websocket("/ws/stream/{national_id}")
async def websocket_stream(websocket: WebSocket, national_id: str):
    """
    Real-time face analysis for the browser webcam.

    The browser sends JSON frames:
        { "frame": "<base64 image data url>", "type": "analysis" | "capture" }

    The server responds with JSON:
        {
          "type": "feedback",
          "face_detected": bool,
          "pose": "front"|"left"|"right"|null,
          "pose_ok": bool,
          "quality_ok": bool,
          "instruction": str,         # shown to the user
          "bounding_box": [x,y,w,h]|null,
          "capture_ready": bool       # true = browser should auto-capture this frame
        }

    For type="capture" frames, the server runs full verification and returns:
        { "type": "result", "decision": str, "score": float, ... }

    Session state tracks:
        - Which poses have been captured (front, left, right)
        - Blink detection across frames (EAR time series)
        - Texture liveness scores
    """
    await websocket.accept()

    # Session state
    session = {
        "captured_poses": {},        # pose -> np.ndarray image
        "texture_scores": [],
        "ear_series": [],            # Eye Aspect Ratio over time (blink detection)
        "blink_confirmed": False,
        "target_pose": "front",      # current requested pose
        "pose_sequence": ["front", "left", "right"],
        "pose_idx": 0,
        "stable_frames": 0,          # frames held in correct pose
        "STABLE_NEEDED": 8,          # frames required before auto-capture
    }

    instructions = {
        "front": "Look straight at the camera",
        "left":  "Slowly turn your head to the left",
        "right": "Slowly turn your head to the right",
    }

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            frame_type = msg.get("type", "analysis")
            b64 = msg.get("frame", "")

            img = decode_frame(b64)
            if img is None:
                await websocket.send_json({"type": "error", "message": "Could not decode frame"})
                continue

            # ── Face detection ──
            emb, face = get_face_embedding_from_array(img)

            if face is None or (isinstance(face, str) and face == "multiple_faces"):
                await websocket.send_json({
                    "type": "feedback",
                    "face_detected": False,
                    "pose": None,
                    "pose_ok": False,
                    "quality_ok": False,
                    "instruction": "No face detected — position your face in the frame",
                    "bounding_box": None,
                    "capture_ready": False,
                })
                continue

            # ── Quality check ──
            quality = _quality_from_array(img)
            quality_ok = quality["score"] >= 7

            # ── Pose classification ──
            pose_info = estimate_head_pose(face)
            detected_pose = classify_pose(pose_info)
            target = session["target_pose"]
            pose_ok = (detected_pose == target)

            # ── Bounding box ──
            bbox = face.bbox.astype(int).tolist() if hasattr(face, "bbox") else None
            bb = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]] if bbox else None

            # ── Texture liveness (every frame) ──
            if bbox:
                x1, y1, x2, y2 = bbox
                crop = img[max(0,y1):y2, max(0,x1):x2]
                tex = assess_texture_liveness(crop)
                session["texture_scores"].append(tex)
                if tex["suspicious"]:
                    await websocket.send_json({
                        "type": "feedback",
                        "face_detected": True,
                        "pose": detected_pose,
                        "pose_ok": False,
                        "quality_ok": quality_ok,
                        "instruction": "Please use a live face — photo detected",
                        "bounding_box": bb,
                        "capture_ready": False,
                        "spoofing_alert": True,
                    })
                    continue

            # ── Stable pose counter ──
            if pose_ok and quality_ok:
                session["stable_frames"] += 1
            else:
                session["stable_frames"] = 0

            capture_ready = session["stable_frames"] >= session["STABLE_NEEDED"]

            instruction = instructions[target]
            if not quality_ok:
                if "blurry" in quality["issues"]:
                    instruction = "Hold still — image is blurry"
                elif "too_dark" in quality["issues"]:
                    instruction = "Move to a brighter area"
                else:
                    instruction = "Move closer to the camera"
            elif not pose_ok:
                instruction = instructions[target]
            elif capture_ready:
                instruction = f"Hold still — capturing {target} pose..."

            # ── Auto-capture triggered ──
            if capture_ready and frame_type != "capture":
                # Save this frame as the captured pose
                session["captured_poses"][target] = img.copy()

                # Advance to next pose
                session["pose_idx"] += 1
                session["stable_frames"] = 0

                if session["pose_idx"] < len(session["pose_sequence"]):
                    session["target_pose"] = session["pose_sequence"][session["pose_idx"]]
                    await websocket.send_json({
                        "type": "pose_captured",
                        "captured_pose": target,
                        "next_pose": session["target_pose"],
                        "instruction": instructions[session["target_pose"]],
                        "poses_done": session["pose_idx"],
                        "poses_total": len(session["pose_sequence"]),
                    })
                    continue
                else:
                    # All poses captured — run full verification
                    await websocket.send_json({
                        "type": "feedback",
                        "face_detected": True,
                        "pose": detected_pose,
                        "pose_ok": True,
                        "quality_ok": True,
                        "instruction": "All poses captured. Verifying...",
                        "bounding_box": bb,
                        "capture_ready": False,
                    })

                    # Save frames to temp files
                    temp_paths = []
                    try:
                        person = get_person_by_national_id(national_id)
                        if not person:
                            await websocket.send_json({"type": "error", "message": "Person not found"})
                            break

                        for pose_name, frame_img in session["captured_poses"].items():
                            p = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{pose_name}.jpg")
                            cv2.imwrite(p, frame_img)
                            temp_paths.append(p)

                        blink_ok = session.get("blink_confirmed", False)

                        result = smart_verify(
                            user_id=person["id"],
                            live_images=temp_paths,
                            attempts=1,
                            known_device=True,
                            blink_detected=blink_ok,
                            texture_scores=session["texture_scores"],
                        )

                        await websocket.send_json({
                            "type": "result",
                            **result,
                        })

                    finally:
                        cleanup(temp_paths)
                    break

            # ── Regular feedback frame ──
            await websocket.send_json({
                "type": "feedback",
                "face_detected": True,
                "pose": detected_pose,
                "pose_ok": pose_ok,
                "quality_ok": quality_ok,
                "yaw": pose_info.get("yaw"),
                "instruction": instruction,
                "bounding_box": bb,
                "capture_ready": capture_ready,
                "stable_frames": session["stable_frames"],
                "stable_needed": session["STABLE_NEEDED"],
                "target_pose": target,
                "blink_confirmed": session["blink_confirmed"],
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
