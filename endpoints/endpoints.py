from pydantic import BaseModel

class PandasCode(BaseModel):
    code: str
    comment: str
