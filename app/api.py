from fastapi import FastAPI, UploadFile, File, HTTPException

from app.schemas import AgenticMeetingRequest, MeetingRequest
from src.agentic import run_agentic_analysis
from src.pipeline import run_pipeline


app = FastAPI(title="Meeting Intelligence API", version="2.0.0")

try:
    import multipart  # noqa: F401

    MULTIPART_AVAILABLE = True
except ImportError:
    MULTIPART_AVAILABLE = False


@app.get("/")
def root() -> dict:
    return {"message": "Meeting Intelligence API is running"}


@app.post("/analyze")
def analyze_meeting(request: MeetingRequest) -> dict:
    return run_pipeline(request.transcript, request.query, top_k=request.top_k)


@app.post("/analyze-agentic")
def analyze_meeting_agentic(request: AgenticMeetingRequest) -> dict:
    return run_agentic_analysis(request.transcript, request.query, top_k=request.top_k)


if MULTIPART_AVAILABLE:

    @app.post("/analyze-file")
    async def analyze_meeting_file(
        file: UploadFile = File(...),
        query: str = "What was decided about the demo?",
        top_k: int = 3,
    ) -> dict:
        transcript = await _read_txt_upload(file)
        return run_pipeline(transcript, query, top_k=top_k)


    @app.post("/analyze-file-agentic")
    async def analyze_meeting_file_agentic(
        file: UploadFile = File(...),
        query: str = "What was decided about the demo?",
        top_k: int = 5,
    ) -> dict:
        transcript = await _read_txt_upload(file)
        return run_agentic_analysis(transcript, query, top_k=top_k)


    async def _read_txt_upload(file: UploadFile) -> str:
        if not file.filename or not file.filename.endswith(".txt"):
            raise HTTPException(status_code=400, detail="Only .txt files are supported.")

        content = await file.read()
        return content.decode("utf-8")

else:

    @app.post("/analyze-file")
    async def analyze_meeting_file_unavailable() -> dict:
        raise HTTPException(
            status_code=503,
            detail="Install python-multipart to enable transcript file uploads.",
        )


    @app.post("/analyze-file-agentic")
    async def analyze_meeting_file_agentic_unavailable() -> dict:
        raise HTTPException(
            status_code=503,
            detail="Install python-multipart to enable transcript file uploads.",
        )
