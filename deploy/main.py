from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from pipeline import run_pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str
    target: int

@app.post("/process")
async def process_text(request: TextRequest):
    json = run_pipeline(request.text, request.target)
    return JSONResponse(
        content=jsonable_encoder(json),
        media_type="application/json",
        
    )
