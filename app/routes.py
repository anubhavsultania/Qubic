from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from .analyzer import analyze_code

router = APIRouter()

@router.post("/analyze/")
async def analyze_cpp_file(file: UploadFile = File(...)):
    if not file.filename.endswith((".cpp", ".h")):
        return JSONResponse({"error": "Only .cpp or .h files allowed"}, status_code=400)

    contents = await file.read()
    cpp_code = contents.decode("utf-8")

    issues = await analyze_code(cpp_code)

    return {"issues": issues}
