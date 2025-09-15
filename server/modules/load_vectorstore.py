import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-index")

UPLOAD_DIR = "./uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)
existing_indexes = [i["name"] for i in pc.list_indexes()]

# HuggingFace embedding model (768-dimensional, no quota issues)
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Create Pinecone index if it doesn't exist
if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,  # Must match embedding dimension of all-mpnet-base-v2
        metric="cosine",
        spec=spec
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)

# Load, split, embed, and upsert PDF content
def load_vectorstore(uploaded_files, batch_size=50):
    file_paths = []

    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))

    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        ids = [f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [{"text": chunk.page_content, **chunk.metadata} for chunk in chunks]

        print(f"🔍 Embedding {len(texts)} chunks...")
        embeddings = embed_model.embed_documents(texts)

        print("📤 Uploading to Pinecone in batches...")
        with tqdm(total=len(embeddings), desc="Upserting to Pinecone") as progress:
            for i in range(0, len(embeddings), batch_size):
                batch_ids = ids[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]

                index.upsert(vectors=zip(batch_ids, batch_embeddings, batch_metadatas))
                progress.update(len(batch_ids))

        print(f"✅ Upload complete for {file_path}")
