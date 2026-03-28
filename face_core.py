import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image
from PIL.ExifTags import TAGS

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))


# ─────────────────────────────────────────────
#  CORE EMBEDDING
# ─────────────────────────────────────────────

def get_face_embedding(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"Cannot read image: {image_path}")
    emb, _ = get_face_embedding_from_array(img)
    if emb is None:
        raise Exception(f"No face detected in {image_path}")
    return emb


def get_face_embedding_from_array(img: np.ndarray):
    """Get embedding + face object directly from a numpy array (real-time use)."""
    faces = app.get(img)
    if len(faces) == 0:
        return None, None
    if len(faces) > 1:
        return None, "multiple_faces"
    embedding = faces[0].embedding.astype(np.float32)
    return embedding / np.linalg.norm(embedding), faces[0]


# ─────────────────────────────────────────────
#  IMAGE QUALITY
# ─────────────────────────────────────────────

def assess_image_quality(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return {"score": 0, "issues": ["invalid_image"]}
    return _quality_from_array(img)


def _quality_from_array(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    h, w = img.shape[:2]

    score = 10
    issues = []
    if blur < 80:
        score -= 3
        issues.append("blurry")
    if brightness < 50:
        score -= 2
        issues.append("too_dark")
    if w < 200 or h < 200:
        score -= 3
        issues.append("low_resolution")

    return {
        "score": max(score, 0),
        "issues": issues,
        "blur": float(blur),
        "brightness": float(brightness),
        "width": int(w),
        "height": int(h),
    }


# ─────────────────────────────────────────────
#  ANTI-SPOOFING: LBP TEXTURE ANALYSIS
# ─────────────────────────────────────────────

def _lbp_map(gray: np.ndarray) -> np.ndarray:
    """Local Binary Pattern — detects micro-texture absent in printed photos."""
    lbp = np.zeros_like(gray, dtype=np.uint8)
    offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    for i, (dy, dx) in enumerate(offsets):
        shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
        lbp += (shifted >= gray).astype(np.uint8) << i
    return lbp


def assess_texture_liveness(face_crop: np.ndarray) -> dict:
    """
    Detect printed/screen photo attacks via LBP texture + frequency analysis.
    Real skin has high-entropy micro-texture; flat photos do not.
    Returns score 0–30.
    """
    if face_crop is None or face_crop.size == 0:
        return {"score": 0, "suspicious": True, "reason": "no_face_crop"}

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))

    # LBP entropy
    lbp = _lbp_map(gray)
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-6)
    entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))

    # Frequency domain: real faces have more high-freq energy
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mag = 20 * np.log(np.abs(f) + 1)
    h, w = mag.shape
    center = mag[h//4:3*h//4, w//4:3*w//4].mean()
    overall = mag.mean()
    freq_ratio = float(overall / (center + 1e-6))

    score = 0
    suspicious = False
    reasons = []

    if entropy > 6.5:
        score += 15
    elif entropy > 5.5:
        score += 8
    else:
        suspicious = True
        reasons.append("low_texture_entropy")

    if freq_ratio > 0.75:
        score += 15
    elif freq_ratio > 0.60:
        score += 8
    else:
        suspicious = True
        reasons.append("low_frequency_variation")

    return {
        "score": min(score, 30),
        "suspicious": suspicious,
        "entropy": round(entropy, 3),
        "freq_ratio": round(freq_ratio, 3),
        "reasons": reasons,
    }


# ─────────────────────────────────────────────
#  ANTI-SPOOFING: HEAD POSE
# ─────────────────────────────────────────────

def estimate_head_pose(face) -> dict:
    """
    Estimate yaw (left/right rotation) from InsightFace face object.
    Uses pose attribute when available, falls back to landmark geometry.
    """
    if hasattr(face, "pose") and face.pose is not None:
        pitch, yaw, roll = face.pose
        return {
            "pitch": round(float(pitch), 1),
            "yaw": round(float(yaw), 1),
            "roll": round(float(roll), 1),
        }

    # Fallback via 5-point landmarks
    lm = face.kps
    left_eye, right_eye, nose = lm[0], lm[1], lm[2]
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    eye_width = abs(right_eye[0] - left_eye[0]) + 1e-6
    yaw_est = float((nose[0] - eye_center_x) / eye_width * 45.0)
    return {"pitch": 0.0, "yaw": round(yaw_est, 1), "roll": 0.0}


def classify_pose(pose: dict) -> str:
    """Returns 'front', 'left', or 'right' based on yaw angle."""
    yaw = pose.get("yaw", 0)
    if yaw < -20:
        return "left"
    if yaw > 20:
        return "right"
    return "front"


# ─────────────────────────────────────────────
#  SCORING
# ─────────────────────────────────────────────

def compare_embeddings(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b)), float(np.linalg.norm(a - b))


def biometric_score(live_embedding, profiles):
    details = []
    for p in profiles:
        stored = np.array(p["embedding"], dtype=np.float32)
        sim, dist = compare_embeddings(live_embedding, stored)
        details.append({"pose": p["pose"], "similarity": round(sim, 4), "distance": round(dist, 4)})

    details.sort(key=lambda x: x["similarity"], reverse=True)
    best_sim = details[0]["similarity"]
    best_dist = details[0]["distance"]
    avg_sim = round(sum(d["similarity"] for d in details) / len(details), 4)
    strong_matches = sum(1 for d in details if d["similarity"] >= 0.60)

    score = 0
    if best_sim >= 0.75:
        score += 30
    elif best_sim >= 0.65:
        score += 20
    elif best_sim >= 0.55:
        score += 10
    if avg_sim >= 0.65:
        score += 15
    elif avg_sim >= 0.55:
        score += 10
    if strong_matches >= 3:
        score += 5

    return {
        "score": score,
        "best_similarity": best_sim,
        "best_distance": best_dist,
        "average_similarity": avg_sim,
        "strong_matches": strong_matches,
        "details": details,
    }


def assess_liveness(embeddings):
    if len(embeddings) < 2:
        return {"score": 0, "reason": "single_capture_only"}
    variation = sum(
        1 for i in range(len(embeddings) - 1)
        if float(np.dot(embeddings[i], embeddings[i + 1])) < 0.95
    )
    return {"score": min(variation * 10, 25), "variation_count": variation}


def assess_device_trust(attempts, known_device):
    score = 15
    flags = []
    if attempts > 5:
        score -= 5
        flags.append("too_many_attempts")
    if not known_device:
        score -= 5
        flags.append("unknown_device")
    return {"score": max(score, 0), "flags": flags}


def read_exif_metadata(image_path: str):
    try:
        img = Image.open(image_path)
        exif = img.getexif()
        if not exif:
            return {"has_exif": False, "fields": {}}
        fields = {str(TAGS.get(tid, tid)): str(v) for tid, v in exif.items()}
        return {"has_exif": True, "fields": fields}
    except Exception:
        return {"has_exif": False, "fields": {}}


def assess_replay_risk(live_embedding, profiles):
    best = max(
        ({"sim": compare_embeddings(live_embedding, np.array(p["embedding"], dtype=np.float32))[0],
          "dist": compare_embeddings(live_embedding, np.array(p["embedding"], dtype=np.float32))[1],
          "pose": p["pose"]} for p in profiles),
        key=lambda x: x["sim"],
    )
    suspicious = best["sim"] >= 0.90 or best["dist"] <= 0.45
    return {
        "suspicious": suspicious,
        "penalty": 20 if suspicious else 0,
        "max_similarity": round(best["sim"], 4),
        "min_distance": round(best["dist"], 4),
        "closest_pose": best["pose"],
        "flags": ["possible_replayed_reference_image"] if suspicious else [],
    }
