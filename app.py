from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title= "RAG Knowledge Assistant")

app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.get("/")
def read_root():
  return{"message": "RAG Knowledge Assistant API"}

@app.post("/upload")
async def upload_file(file:UploadFile = File(...)):
  return {"filename":file.filename, "status": "uploaded"}
  
