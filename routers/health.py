from fastapi import APIRouter
import datetime

router = APIRouter(prefix="/health", tags=["System"])


@router.get("/")
def health_check():
    return {"status": "ok", "time": datetime.datetime.utcnow().isoformat()}
