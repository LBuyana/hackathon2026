import os
import numpy as np
from dotenv import load_dotenv
from supabase import create_client

from face_core import (
    get_face_embedding,
    assess_image_quality,
    biometric_score,
    assess_liveness,
    assess_device_trust,
    assess_replay_risk,
    assess_texture_liveness,
    read_exif_metadata,
)

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY"),
)


# ─────────────────────────────────────────────
#  PERSON MANAGEMENT
# ─────────────────────────────────────────────

def create_person(national_id, first_name, last_name,
                  date_of_birth=None, gender=None,
                  phone=None, email=None, address=None):
    result = (
        supabase.table("persons")
        .insert({
            "national_id": national_id,
            "first_name": first_name,
            "last_name": last_name,
            "date_of_birth": date_of_birth,
            "gender": gender,
            "phone": phone,
            "email": email,
            "address": address,
        })
        .execute()
    )
    return result.data[0]


def get_person_by_national_id(national_id):
    result = (
        supabase.table("persons")
        .select("*")
        .eq("national_id", national_id)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


# ─────────────────────────────────────────────
#  FACE GALLERY
# ─────────────────────────────────────────────

def get_face_profiles(user_id):
    res = (
        supabase.table("face_profiles")
        .select("*")
        .eq("user_id", user_id)
        .execute()
    )
    return res.data


def enroll_reference_gallery(user_id: str, gallery: dict):
    """
    Enrol reference face images for a person.
    gallery = {"front": "/path/img.jpg", "left": "...", "right": "..."}
    """
    inserted = []
    for pose, image_path in gallery.items():
        quality = assess_image_quality(image_path)
        if quality["score"] == 0:
            raise Exception(f"Invalid image for pose '{pose}': {image_path}")

        embedding = get_face_embedding(image_path)

        record = (
            supabase.table("face_profiles")
            .insert({
                "user_id": user_id,
                "pose": pose,
                "embedding": embedding.tolist(),
                "quality_score": quality["score"],
                "quality_issues": quality["issues"],
            })
            .execute()
        )
        inserted.append(record.data[0])
    return inserted


# ─────────────────────────────────────────────
#  VERIFICATION
# ─────────────────────────────────────────────

def smart_verify(user_id, live_images, attempts=1, known_device=True,
                 blink_detected=False, texture_scores=None):
    """
    Full biometric verification pipeline.

    Args:
        user_id:          Person's DB id (from persons table)
        live_images:      List of image file paths (front required, left/right optional)
        attempts:         Number of verification attempts this session
        known_device:     Whether device fingerprint is recognised
        blink_detected:   Whether the browser confirmed a blink during capture
        texture_scores:   Optional list of texture liveness dicts (from browser frames)
    """
    profiles = get_face_profiles(user_id)
    if not profiles:
        raise Exception("No enrolled gallery found for this user")

    total_quality = 0
    embeddings = []
    bio_results = []
    quality_reports = []
    replay_reports = []
    metadata_reports = []
    texture_reports = texture_scores or []

    for img_path in live_images:
        quality = assess_image_quality(img_path)
        quality_reports.append({"image": img_path, **quality})
        total_quality += quality["score"]

        metadata = read_exif_metadata(img_path)
        metadata_reports.append({"image": img_path, **metadata})

        emb = get_face_embedding(img_path)
        embeddings.append(emb)

        bio = biometric_score(emb, profiles)
        bio_results.append(bio)

        replay = assess_replay_risk(emb, profiles)
        replay_reports.append({"image": img_path, **replay})

    best_bio = max(bio_results, key=lambda x: x["score"])
    liveness = assess_liveness(embeddings)
    device = assess_device_trust(attempts, known_device)
    avg_quality = round(total_quality / len(live_images), 2)
    replay_penalty = sum(r["penalty"] for r in replay_reports)

    # Anti-spoofing bonuses
    spoofing_penalty = 0
    spoofing_flags = []

    if not blink_detected:
        spoofing_penalty += 15
        spoofing_flags.append("no_blink_detected")

    if texture_reports:
        avg_texture = sum(t.get("score", 0) for t in texture_reports) / len(texture_reports)
        texture_suspicious = any(t.get("suspicious", False) for t in texture_reports)
        if texture_suspicious:
            spoofing_penalty += 20
            spoofing_flags.append("texture_analysis_failed")
    else:
        avg_texture = 0

    total_score = (
        best_bio["score"]
        + liveness["score"]
        + device["score"]
        + avg_quality
        + avg_texture
        - replay_penalty
        - spoofing_penalty
    )

    suspicious_replay = any(r["suspicious"] for r in replay_reports)
    single_capture = len(live_images) < 2

    if suspicious_replay and single_capture:
        decision = "Manual Review"
    elif spoofing_flags:
        decision = "Manual Review"
    elif total_score >= 80:
        decision = "Express Approved"
    elif total_score >= 55:
        decision = "Manual Review"
    else:
        decision = "More Information Needed"

    return {
        "score": round(total_score, 2),
        "decision": decision,
        "biometric": best_bio,
        "liveness": liveness,
        "device": device,
        "quality": {"average_score": avg_quality, "reports": quality_reports},
        "replay_risk": replay_reports,
        "anti_spoofing": {
            "blink_detected": blink_detected,
            "texture_score": round(avg_texture, 2),
            "penalty": spoofing_penalty,
            "flags": spoofing_flags,
        },
        "metadata": metadata_reports,
    }
