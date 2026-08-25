"""Temporary signed-session guard for opening the annotation workbench from AIS."""

import hashlib
import hmac
import os
import time

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/annotation-session", tags=["annotation-session"])
_SECRET = os.environ.get("ANNOTATION_TOKEN_SECRET")
if not _SECRET:
    if os.environ.get("AIS_DEV_MODE") == "1":
        _SECRET = "ais-annotation-development-secret"
    else:
        raise RuntimeError("ANNOTATION_TOKEN_SECRET must be configured outside development.")
_SECRET = _SECRET.encode()


def _signature(payload: str) -> str:
    return hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()


@router.get("/validate")
def validate_session(
    token: str = Query(...), report_id: str = Query(..., alias="reportId"), subject_id: str = Query(..., alias="subjectId"),
) -> dict:
    try:
        payload, signature = token.rsplit(".", 1)
        token_report, token_subject, expires_at = payload.rsplit(".", 2)
        valid = hmac.compare_digest(signature, _signature(payload))
        if not valid or token_report != report_id or token_subject != subject_id or int(expires_at) < int(time.time() * 1000):
            raise ValueError
    except (ValueError, TypeError):
        raise HTTPException(status_code=403, detail="标注授权无效或已过期")
    return {"reportId": token_report, "subjectId": token_subject, "expiresAt": int(expires_at)}
