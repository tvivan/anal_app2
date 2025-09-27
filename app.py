import os
import uuid
import tempfile
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from functions import df_code_analys, api_integration, prompts
from functions.memory import MemoryManager
from endpoints.endpoints import (
    PandasCode,
    ChatRequest,
    ChatResponse,
    StateResponse,
    UploadResponse
)

load_dotenv()
app = FastAPI(
    title="CSV Analyst API",
    description="API для интерактивного анализа CSV файлов с помощью LLM.",
    version="1.0.0"
)

# Создаем синглтоны менеджера памяти и LLM клиента. Живут, пока жива сессия
try:
    mgr = MemoryManager(redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    client = api_integration.LLMClient(provider_name="openrouter")
except ValueError as e:
    print(f"КРИТИЧЕСКАЯ ОШИБКА при инициализации: {e}")
    mgr = None
    client = None

# Эндпоинты
@app.post("/upload", response_model=UploadResponse, tags=["Session"])
async def upload_csv_and_start_session(file: UploadFile = File(...)):
    """Creating session with csv

    Args:
        file (UploadFile, optional): upload file method from fastapi. Defaults to File(...).

    Raises:
        HTTPException: exeption for only CSV
        HTTPException: redis not working

    Returns:
        _type_: upload session id, file and meta data
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Поддерживаются только CSV файлы.")
    if not mgr:
        raise HTTPException(status_code=503, detail="Сервис MemoryManager не инициализирован. Проверьте конфигурацию Redis.")

    session_id = str(uuid.uuid4())


    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        # Инициализируем сессию с помощью MemoryManager
        meta = mgr.init_session_from_csv(session_id, temp_file_path)
    finally:
        # Удаляем временный файл после использования
        os.remove(temp_file_path)

    return UploadResponse(
        session_id=session_id,
        filename=file.filename,
        initial_meta=meta
    )

@app.post("/chat/{session_id}", response_model=ChatResponse, tags=["Analysis"])
def handle_chat_query(
    request: ChatRequest,
    session_id: str = Path(..., description="ID сессии, полученный после загрузки файла")
):
    """Function for handling chat query via redis

    Args:
        request (ChatRequest): pydantic class from endpoints
        session_id (str, optional): session id (..., description="ID сессии, полученный после загрузки файла").

    Returns:
        _type_: dictionary with new meanings after processing for meta data
    """
    if not mgr or not client:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован. Проверьте конфигурацию.")

    df = mgr.load_current_df(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Сессия с ID '{session_id}' не найдена.")

    info = df_code_analys.pd_getinfo(df)

    system_prompt = prompts.prompt_code_generation(info=info, querry=request.query)

    try:
        response = client.chat.completions.parse(
            model="qwen/qwen3-30b-a3b:free",
            messages=system_prompt,
            response_format=PandasCode,  # Используем импортированную модель
            temperature=0,
            top_p=0.95
        )
        parsed_result = response.choices[0].message.parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обращении к LLM: {e}")

    execution_result = df_code_analys.normalize_and_execute_code(parsed_result.code, df)

    final_meta = mgr.get_current_state_info(session_id) # Метаданные по умолчанию

    if isinstance(execution_result, pd.DataFrame):
        # Если код вернул новый DataFrame, сохраняем его как новое состояние
        final_meta = mgr.push_result(session_id, execution_result, code=parsed_result.code)
        df_preview_html = execution_result.head().to_html(classes='table table-striped', justify='left')
        return ChatResponse(
            session_id=session_id,
            comment=parsed_result.comment,
            code=parsed_result.code,
            df_preview_html=df_preview_html,
            meta=final_meta
        )
    else:
        # Если результат не DataFrame, состояние не меняется. Просто возвращаем результат вычислений.
        current_df = mgr.load_current_df(session_id)
        df_preview_html = current_df.head().to_html(classes='table table-striped', justify='left')
        return ChatResponse(
            session_id=session_id,
            comment=parsed_result.comment,
            code=parsed_result.code,
            execution_result=str(execution_result), # Преобразуем в строку для JSON-совместимости
            df_preview_html=df_preview_html,
            meta=final_meta
        )


@app.post("/undo/{session_id}", response_model=StateResponse, tags=["State Management"])
def undo_last_action(session_id: str = Path(..., description="ID сессии")):
    """undo action

    Args:
        session_id (str, optional): session id

    Returns:
        _type_: new meta data for redis
    """
    if not mgr:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован.")

    meta = mgr.undo(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Сессия '{session_id}' не найдена или нет состояний для отмены.")

    df = mgr.load_current_df(session_id)
    return StateResponse(
        session_id=session_id,
        meta=meta,
        df_preview_html=df.head().to_html(classes='table table-striped', justify='left')
    )


@app.post("/redo/{session_id}", response_model=StateResponse, tags=["State Management"])
def redo_last_action(session_id: str = Path(..., description="ID сессии")):
    if not mgr:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован.")

    meta = mgr.redo(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Сессия '{session_id}' не найдена или нет состояний для повтора.")

    df = mgr.load_current_df(session_id)
    return StateResponse(
        session_id=session_id,
        meta=meta,
        df_preview_html=df.head().to_html(classes='table table-striped', justify='left')
    )

@app.get("/state/{session_id}", response_model=StateResponse, tags=["State Management"])
def get_current_state(session_id: str = Path(..., description="ID сессии")):
    """takes current state from redis

    Args:
        session_id (str, optional): id session

    Returns:
        _type_: meta data for redis
    """
    if not mgr:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован.")

    meta = mgr.get_current_state_info(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Сессия '{session_id}' не найдена.")

    df = mgr.load_current_df(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Не удалось загрузить DataFrame для сессии '{session_id}'.")

    return StateResponse(
        session_id=session_id,
        meta=meta,
        df_preview_html=df.head().to_html(classes='table table-striped', justify='left')
    )