import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.rag import rag_engine


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting application. Thanks for using pdHelp! Follow @joshmode on GitHub for more")
    rag_engine.initialize()
    if rag_engine.vector_store is None or rag_engine.llm is None:
        raise RuntimeError("initialization failed. check logs for details.")
    yield


app = FastAPI(title="pdhelp by @joshmode", description="a tool to help with pdfs.", lifespan=lifespan)


class QueryRequest(BaseModel):
    text: Optional[str] = None
    question: Optional[str] = None
    query: Optional[str] = None


class QueryResponse(BaseModel):
    reply: str


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="only pdf files are supported.")

    temp_file_path = None
    try:
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)

        if file_size == 0:
            raise HTTPException(status_code=400, detail="uploaded file is empty.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        chunks = rag_engine.process_document(temp_file_path)
        if not chunks:
            raise HTTPException(status_code=400, detail="document appears to be empty or unreadable.")

        rag_engine.add_documents(chunks)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error processing file: %s", temp_file_path)
        raise HTTPException(status_code=500, detail=f"error processing file: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return {"message": "document processed successfully", "filename": file.filename}


@app.post("/query", response_model=QueryResponse)
def query_llm(request: QueryRequest):
    prompt = (request.text or request.question or request.query or "").strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="please provide a question.")

    try:
        answer = rag_engine.query(prompt)
        return QueryResponse(reply=answer)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error generating answer for prompt: %s", prompt)
        raise HTTPException(status_code=500, detail=f"error generating answer: {str(e)}")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "application is running",
    }


@app.get("/")
def serve_frontend():
    return FileResponse("app/static/index.html")
