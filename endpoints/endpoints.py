from pydantic import BaseModel
from typing import Optional, Any, Dict

# Модель для LLM
class PandasCode(BaseModel):
    code: str
    comment: str

# Модели для эндпоинтов API

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    session_id: str
    comment: str
    code: str
    execution_result: Optional[Any] = None
    df_preview_html: Optional[str] = None
    meta: Dict[str, Any]

class StateResponse(BaseModel):
    session_id: str
    meta: Dict[str, Any]
    df_preview_html: str

class UploadResponse(BaseModel):
    session_id: str
    filename: str
    initial_meta: Dict[str, Any]