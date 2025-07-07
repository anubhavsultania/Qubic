from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)

app = FastAPI(
    title="C++ Code Auditor API",
    description="Audits C++ code using RAG-based LLM",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
