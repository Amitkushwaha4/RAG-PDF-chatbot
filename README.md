### 1. Setup the Backend (FastAPI)

```bash
cd server
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt

# Set your Groq API Key (.env)
GROQ_API_KEY="your_key_here"

# Run the FastAPI server
uvicorn main:app --reload
```

### 2. Setup the Frontend (Streamlit)

```bash
cd ../client
pip install -r requirements.txt  # if you use a separate venv for client
streamlit run app.py
```

---

## 🌐 API Endpoints (FastAPI)

- `POST /upload_pdfs/` — Upload PDFs and build vectorstore
- `POST /ask/` — Send a query and receive answers

Testable via Postman or directly from the Streamlit frontend.