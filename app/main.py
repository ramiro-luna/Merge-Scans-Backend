from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router

app = FastAPI(title="Scan Merge API")

# 15MB limit (in bytes)
MAX_UPLOAD_SIZE = 15 * 1024 * 1024

@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    # Check Content-Length header
    content_length = request.headers.get("Content-Length")
    if content_length and int(content_length) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large (Max: 15MB)")
    
    return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Scan Merge API is running"}

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "scan-merge-api"}

app.include_router(router)