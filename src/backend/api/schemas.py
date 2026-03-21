from pydantic import BaseModel
from typing import Optional


class WorkspaceCreateResponse(BaseModel):
    workspace_id: str


class DocumentUploadResponse(BaseModel):
    workspace_id: str
    file_name: str
    status: str
    chunk_count: int


class ResearchRequest(BaseModel):
    goal: str
    workspace_id: str


class ResearchStartResponse(BaseModel):
    workspace_id: str
    task_id: str

class ReviewRequest(BaseModel):
    task_id: str
    workspace_id: str
    action: str
    feedback: Optional[str] = None
