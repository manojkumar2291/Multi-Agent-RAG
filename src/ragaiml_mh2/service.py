"""The two main flows in this project.

Upload: file bytes -> page text -> small chunks -> embeddings stored in Chroma.
Question: plan a search -> retrieve chunks -> write an answer -> review it.
"""

import hashlib
import io
from pathlib import PurePosixPath
from threading import RLock
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from .agents import AnswerAgent, QueryPlannerAgent, RetrievalAgent, ReviewAgent
from .config import Settings
from .model import create_chat_model, create_vector_store


class RAGService:
    """Manage file processing, Chroma storage and the four agent roles.

    Created in api.py: get_service(), then reused for later requests.
    Used in api.py: upload_files() calls ingest(); ask_question() calls answer().
    main.py exposes the FastAPI app; API routes call this service.
    Model and database setup functions are defined in model.py.
    Tests create it with a temporary database and a fake model in test_rag.py.
    """

    def __init__(self, settings: Settings, vector_store=None, model=None):
        # Tests can supply a small test database and a fake model.
        # Normal requests use the setup functions in model.py.
        if vector_store is None:
            vector_store = create_vector_store(settings)
        if model is None:
            model = create_chat_model(settings)
        self.settings = settings
        self.store = vector_store
        # Only one upload at a time may check for duplicates and write chunks.
        self.lock = RLock()
        # Neighboring chunks share some text so a sentence keeps its context.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        # The language agents share one model but each has a different job.
        self.planner = QueryPlannerAgent(model)
        self.retriever = RetrievalAgent(vector_store)
        self.writer = AnswerAgent(model)
        self.reviewer = ReviewAgent(model)

    def ingest(self, filename: str, data: bytes) -> dict:
        """Read one uploaded file and save its text chunks in Chroma."""
        # Step 1: Check the name, type and size of the file.
        # Keep only the name, even if the client sends something like ../notes.txt.
        filename = PurePosixPath(filename.replace("\\", "/")).name
        suffix = PurePosixPath(filename).suffix.lower()
        if suffix not in {".pdf", ".txt", ".md"}:
            raise ValueError("Supported files: PDF, UTF-8 TXT and Markdown")
        if not data:
            raise ValueError("The uploaded file is empty")
        if len(data) > self.settings.max_upload_mb * 1024 * 1024:
            raise ValueError("File exceeds the upload size limit")
        # The same file bytes always produce the same ID, even if renamed.
        document_id = hashlib.sha256(data).hexdigest()
        # Step 2: Read the file into (page number, page text) pairs.
        if suffix == ".pdf":
            try:
                # BytesIO lets the PDF reader treat uploaded bytes like a file.
                reader = PdfReader(io.BytesIO(data))
                if reader.is_encrypted:
                    raise ValueError("Encrypted PDFs are not supported")
                pages = []
                for page_number, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    pages.append((page_number, text))
            except Exception as exc:
                raise ValueError("Cannot read PDF; provide an unencrypted, text-based PDF") from exc
        else:
            try:
                # Text and Markdown have no PDF page numbers, so use page 1.
                text = data.decode("utf-8-sig")
                pages = [(1, text)]
            except UnicodeDecodeError as exc:
                raise ValueError("Text files must use UTF-8 encoding") from exc
        # Step 3: Attach source details to each nonempty page.
        # LangChain's Document holds text plus a metadata dictionary.
        documents = []
        for page_number, text in pages:
            if not text.strip():
                continue
            document = Document(
                page_content=text,
                metadata={
                    "document_id": document_id,
                    "filename": filename,
                    "page": page_number,
                },
            )
            documents.append(document)

        # Step 4: Split long pages into smaller pieces for searching.
        chunks = self.splitter.split_documents(documents)
        if not chunks:
            raise ValueError("No extractable text found; scanned PDFs require OCR first")
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk"] = i
        with self.lock:
            # Step 5: Skip a file if its ID is already in Chroma.
            existing = self.store.get(where={"document_id": document_id}, limit=1)
            if existing["ids"]:
                return {
                    "document_id": document_id,
                    "filename": filename,
                    "chunks": 0,
                    "status": "already_indexed",
                }
            # Each chunk needs its own ID: file ID followed by chunk number.
            ids = []
            for chunk_number in range(len(chunks)):
                ids.append(f"{document_id}:{chunk_number}")
            try:
                # Step 6: Chroma uses the embedding model to turn text into
                # vectors, then saves the vectors, original text and metadata.
                # Small batches avoid sending too much text at once.
                for start in range(0, len(chunks), 64):
                    batch_chunks = chunks[start:start + 64]
                    batch_ids = ids[start:start + 64]
                    self.store.add_documents(batch_chunks, ids=batch_ids)
            except Exception:
                # Remove any partial upload if one of the batches fails.
                self.store.delete(ids=ids)
                raise
        return {
            "document_id": document_id,
            "filename": filename,
            "chunks": len(chunks),
            "status": "indexed",
        }

    def answer(self, question: str, k: int = 5, document_ids: list[str] | None = None) -> dict:
        """Pass a question through the four agent roles and return its sources."""
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be blank")
        # First check for stored text without calling the language model.
        # Chroma's $in means the document ID must be in the supplied list.
        if document_ids:
            stored = self.store.get(
                limit=1, where={"document_id": {"$in": document_ids}}
            )
        else:
            stored = self.store.get(limit=1)

        if not stored["ids"]:
            return {
                "answer": "No documents found. Upload files before asking a question.",
                "sources": [],
                "search_query": question,
                "agents": [],
            }
        # Agent 1: Rewrite the question into a search query.
        query = self.planner.run(question)
        query = query[:2000]
        if not query:
            query = question

        # Agent 2: Retrieve at most k chunks with similar meaning.
        documents = self.retriever.run(query, k, document_ids)

        # Give every chunk a citation label and include its original text.
        sources = []
        for source_number, document in enumerate(documents, start=1):
            source = {
                "label": f"Source {source_number}",
                "document_id": document.metadata["document_id"],
                "filename": document.metadata["filename"],
                "page": document.metadata["page"],
                "excerpt": document.page_content,
            }
            sources.append(source)
        if not sources:
            return {
                "answer": "No relevant document excerpts found.",
                "sources": [],
                "search_query": query,
                "agents": ["query_planner", "retriever"],
            }
        # Context is the retrieved text given to the model as evidence.
        context_parts = []
        for source in sources:
            heading = f"[{source['label']}] {source['filename']}, page {source['page']}"
            context_parts.append(heading + "\n" + source["excerpt"])
        context = "\n\n".join(context_parts)
        task = f"Question:\n{question}\n\nDocument excerpts:\n{context}"

        # Agent 3: Write a draft using the retrieved evidence.
        draft = self.writer.run(task)

        # Agent 4: Check the draft using the same evidence and return a revision.
        review_task = f"{task}\n\nDraft answer:\n{draft}"
        answer = self.reviewer.run(review_task)
        if not answer:
            raise RuntimeError("The review agent returned an empty answer")
        return {
            "answer": answer,
            "sources": sources,
            "search_query": query,
            "agents": ["query_planner", "retriever", "answer_writer", "reviewer"],
        }
