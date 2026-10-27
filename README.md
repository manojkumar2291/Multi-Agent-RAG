# Multi-agent RAG with FastAPI and Chroma

Upload PDF, UTF-8 TXT or Markdown files, then ask questions grounded in their
contents. Embeddings persist in Chroma across restarts. The default language
model runs locally through Ollama; no OpenAI API key is required.

Use FastAPI's built-in **http://127.0.0.1:8000/docs** page to upload files and ask
questions. For a beginner walkthrough of every project file, class, function,
and request flow, read [CODE_EXPLAINED.md](CODE_EXPLAINED.md).

The coordinator runs four specialist roles in sequence:

1. **Query planner** rewrites the question for semantic search.
2. **Retrieval agent** searches Chroma, optionally restricted to document IDs.
3. **Answer agent** writes an answer from retrieved excerpts with source labels.
4. **Review agent** checks and revises the draft against the same excerpts.

The planner, writer and reviewer are LangChain agents built with `create_agent()`.
Each has its own `system_prompt` and makes a separate model call. The retrieval
step searches Chroma directly. This is a fixed workflow,
without autonomous tool selection. Review reduces unsupported claims but does
not guarantee correctness; returned excerpts let callers check the evidence.

## Where to start reading the code

Read these files in this order:

1. `main.py`: gives Uvicorn the FastAPI application to run.
2. `src/ragaiml_mh2/api.py`: receives uploaded files and questions. Start with
   `upload_files()` and `ask_question()`. They import the Pydantic data models
   from `src/ragaiml_mh2/models.py`, which defines the JSON fields and validation.
3. `src/ragaiml_mh2/service.py`: `ingest()` handles uploads step by step;
   `answer()` passes a question through the four agent roles.
4. `src/ragaiml_mh2/agents.py`: each class represents one job. Its `run()` method
   does that job. Each language agent is created with `create_agent(model=...,
   tools=[], system_prompt=...)`. The shared `run_agent()` function calls
   `agent.invoke()` and reads the final reply from `result["messages"]`.
5. `src/ragaiml_mh2/config.py`: model names, database location and upload limits.
6. `src/ragaiml_mh2/model.py`: creates the Ollama model, embedding model and
   Chroma store. The service calls these setup functions when needed.
7. `tests/test_rag.py`: examples of calling the API and checking its results.

For an upload, the service reads the text, splits it into chunks, and asks Chroma
to store them. Chroma uses an embedding model to convert each chunk into a list
of numbers representing its meaning. It saves those numbers together with the
text and source details.

For a question, the planner creates a search query. Chroma compares the query's
embedding with the saved embeddings to find similar chunks. The writer receives
those chunks as context, then the reviewer checks the draft against the same
chunks. The API returns the reviewed answer and the source excerpts.

`Depends(service_dependency)` in an API route tells FastAPI how to get the RAG
service. `@lru_cache(maxsize=1)` keeps the created service in memory so the next
request can reuse it. `self` inside a class refers to the current object; for
example, `self.store` is the Chroma database used by that service object.

## Setup

Use Python 3.12 or later. In PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
if (!(Test-Path .env)) { Copy-Item .env.example .env }
ollama pull qwen2.5:3b
```

Keep any existing `.env` values when adding the example settings. Start Ollama
with `ollama serve` if the Ollama application is not already running. Then:

```powershell
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000/docs** to upload files and ask questions.
The embedding model downloads from Hugging Face on the first upload or query,
which requires internet access and may take a few minutes. API import and
`/health` do not initialize models. The installed `ragaiml-mh2` command also
starts the API.

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /upload` | Multipart `files` field, one or multiple files |
| `POST /ask` | JSON question, optional `k` and `document_ids` |
| `GET /answer?Question=...` | Compatibility query endpoint with the same response schema |
| `GET /health` | Process liveness; does not check model availability |

```powershell
curl.exe -X POST http://127.0.0.1:8000/upload -F "files=@AST-1.pdf" -F "files=@AST-2.pdf"

$question = @{ question = "What is software testing?"; k = 5 } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/ask -Method Post -ContentType application/json -Body $question
```

Upload responses contain a `document_id`, filename, chunk count and status.
Sending identical bytes again returns `already_indexed`, retaining the original
filename metadata. Ask responses contain `answer`, `sources` (filename, one-based
page, document ID, excerpt and citation label), `search_query` and the agent roles
that ran. To search only selected files, pass their upload IDs as `document_ids`.

Uploads allow up to 10 files, each at most 20 MB by default. Invalid files return
400, oversized files 413, invalid questions 422 and backend failures 503. Files
are processed sequentially; earlier successfully indexed files remain if a later
file in the request fails. Original uploads are not saved, only extracted chunks
and metadata. Scanned PDFs require external OCR; encrypted PDFs are rejected.

Configuration lives in `.env`; see `.env.example`. Use a new Chroma collection
when changing embedding models or chunk settings, and upload files again. The
default `ragaiml_multiagent` collection is separate from the old prototype's
`ragaiml` collection. Run one API worker for this local embedded database setup.
This local API has no authentication or user isolation; add both before sharing
access to private documents.

## Verification

```powershell
python -m unittest discover -s tests -v
```

Tests use real temporary Chroma databases, deterministic test embeddings and a
fake language model. They cover uploads, duplicate handling, persistence,
document filtering, all agent stages, validation and backend failure handling
without downloading models or requiring Ollama. A live model smoke test requires
the models above: upload a document and ask a question through `/docs`.

Implementation references: [FastAPI file uploads](https://fastapi.tiangolo.com/tutorial/request-files/)
and [LangChain integrations](https://reference.langchain.com/python/integrations/overview).
