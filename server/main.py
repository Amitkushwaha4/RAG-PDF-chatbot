from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from modules.load_vectorstore import load_vectorstore
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from langchain_core.documents import Document
from langchain.schema import BaseRetriever
from pydantic import Field
from logger import logger
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI(title="RagBot")

load_dotenv() 

# Allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.middleware("http")
async def catch_exception_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.exception("UNHANDLED EXCEPTION")
        return JSONResponse(status_code=500, content={"error": str(exc)})

@app.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    try:
        logger.info(f"Received {len(files)} files")
        load_vectorstore(files)
        logger.info("Documents added to Pinecone")
        return {"message": "Files processed and vectorstore updated"}
    except Exception as e:
        logger.exception("Error during PDF upload")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"user query: {question}")

        # 1. Pinecone setup
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

        # 2. HuggingFace embeddings (local, matches PDF upload embeddings)
        embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # 3. Embed the query
        embedded_query = embed_model.embed_query(question)

        # 4. Query Pinecone
        res = index.query(vector=embedded_query, top_k=3, include_metadata=True)

        # 5. Convert to LangChain Documents
        docs = [
            Document(
                page_content=match["metadata"].get("text", ""),
                metadata=match["metadata"]
            ) for match in res["matches"]
        ]

        # 6. Simple retriever wrapper
        class SimpleRetriever(BaseRetriever):
            tags: Optional[List[str]] = Field(default_factory=list)
            metadata: Optional[dict] = Field(default_factory=dict)

            def __init__(self, documents: List[Document]):
                super().__init__()
                self._docs = documents

            def _get_relevant_documents(self, query: str) -> List[Document]:
                return self._docs

        retriever = SimpleRetriever(docs)

        # 7. LLM + RetrievalQA chain
        chain = get_llm_chain(retriever)
        result = query_chain(chain, question)

        logger.info("Query successful")
        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
async def test():
    return {"message": "Testing successful..."}
