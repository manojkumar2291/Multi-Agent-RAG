"""Pydantic models describe and validate the API's JSON data.

api.py imports these classes for its request and response schemas.
These are data models, separate from the AI model setup in model.py.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# BaseModel describes the JSON fields FastAPI should accept or return.
# FastAPI checks request fields before a route and response fields afterward.
class QuestionRequest(BaseModel):
    """Describe and validate a question, chunk count and optional file IDs.

    Used by ask_question() in api.py: FastAPI builds it from POST /ask JSON.
    Also created by legacy_answer() in api.py to handle GET /answer with the same rules.
    Its fields are passed to RAGService.answer() in service.py.
    """

    question: str = Field(min_length=1, max_length=4000)
    # k is the maximum number of text chunks to retrieve (between 1 and 20).
    k: int = Field(default=5, ge=1, le=20)
    # None means search all files. Supply upload IDs to search selected files.
    document_ids: list[str] | None = Field(default=None, min_length=1, max_length=100)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value):
        # min_length alone would allow a question containing only spaces.
        if not value.strip():
            raise ValueError("Question cannot be blank")
        return value.strip()


class Source(BaseModel):
    """Describe one text excerpt and the file and page it came from.

    Used by AnswerResponse below as the type of each item in sources.
    RAGService.answer() in service.py builds source dictionaries; FastAPI checks
    those dictionaries against this class when returning /ask or /answer JSON.
    """

    label: str
    document_id: str
    filename: str
    page: int
    excerpt: str


class AnswerResponse(BaseModel):
    """Describe the final answer, source excerpts, search query and agent names.

    Used as response_model on ask_question() and legacy_answer() in api.py.
    FastAPI uses it to check and format the dictionary from RAGService.answer().
    It also documents the response fields on the /docs page.
    """

    answer: str
    sources: list[Source]
    search_query: str
    agents: list[str]


class UploadResult(BaseModel):
    """Describe one file's ID, name, chunk count and indexing status.

    Used by UploadResponse below as the type of each item in files.
    RAGService.ingest() in service.py returns the dictionary matching these fields.
    FastAPI checks it when upload_files() returns the upload response.
    """

    document_id: str
    filename: str
    chunks: int
    status: Literal["indexed", "already_indexed"]


class UploadResponse(BaseModel):
    """Group the results of all files in one upload request.

    Used as response_model on upload_files() in api.py for POST /upload.
    That route collects the results of RAGService.ingest() into the files list.
    FastAPI uses this class to check and document the returned JSON.
    """

    files: list[UploadResult]
