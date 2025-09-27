import os
import uuid
import tempfile
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from functions.memory import MemoryManager
from functions import df_code_analys, api_integration, prompts
from endpoints.endpoints import PandasCode

st.set_page_config(layout="wide", page_title="Chat with CSV")
load_dotenv()


@st.cache_data
def load_providers():
    """Загружает список провайдеров из api.json."""
    try:
        with open("api.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Файл api.json не найден!")
        return {}

providers_config = load_providers()
provider_names = list(providers_config.keys())

# Инициализация состояния приложения
if "services_initialized" not in st.session_state:
    st.session_state.services_initialized = False
    st.session_state.mgr = None
    st.session_state.client = None
    st.session_state.session_id = None
    st.session_state.messages = []
    st.session_state.current_df = None
    st.session_state.last_result = None


# 1. Левая колонка (сайдбар) для настроек и загрузки файла
with st.sidebar:
    st.title("Настройки и управление")

    # Блок настроек подключения
    with st.expander("⚙️ Настройки подключения", expanded=not st.session_state.services_initialized):
        st.info("Введите данные для подключения и нажмите 'Применить'.")
        redis_url = st.text_input("URL для Redis", value=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        provider_name = st.selectbox("Провайдер LLM", options=provider_names, index=0)
        model_name = st.text_input("Модель", value="qwen/qwen3-30b-a3b:free")
        api_key = st.text_input("API Ключ (если требуется)", type="password")

        if st.button("Применить и подключиться"):
            with st.spinner("Проверка подключений..."):
                try:
                    # Инициализируем MemoryManager
                    mgr = MemoryManager(redis_url=redis_url)
                    mgr.r.ping() # Проверяем соединение с Redis
                    st.session_state.mgr = mgr

                    # Инициализируем LLMClient
                    client = api_integration.LLMClient(provider_name=provider_name, api_key=api_key)
                    st.session_state.client = client

                    # Сохраняем настройки
                    st.session_state.services_initialized = True
                    st.session_state.model_name = model_name
                    st.success("Подключения успешно настроены!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка подключения: {e}")

    st.divider()

    # Блок загрузки файла (активен только после настройки)
    st.title("1. Загрузка CSV")
    uploaded_file = st.file_uploader(
        "Выберите CSV файл", 
        type="csv", 
        disabled=not st.session_state.services_initialized
    )

    if uploaded_file and st.session_state.session_id is None:
        if st.button("Начать сессию"):
            with st.spinner("Создание сессии..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                session_id = str(uuid.uuid4())
                st.session_state.mgr.init_session_from_csv(session_id, tmp_path)
                os.remove(tmp_path)

                st.session_state.session_id = session_id
                st.session_state.current_df = st.session_state.mgr.load_current_df(session_id)
                st.session_state.messages = [{"role": "assistant", "content": f"Сессия `{session_id}` начата. Можете задавать вопросы!"}]
                st.session_state.last_result = None
                st.success("Сессия создана!")
                st.rerun()


# --- Основная часть экрана ---
if not st.session_state.services_initialized:
    st.info("⬅️ Пожалуйста, настройте подключения в боковой панели, чтобы начать работу.")
else:
    col_main, col_chat = st.columns([2, 1])
    
    # 2.1. Центральная колонка
    with col_main:
        st.header("Анализ данных")
        if st.session_state.current_df is not None:
            st.dataframe(st.session_state.current_df, use_container_width=True)
            if st.session_state.last_result:
                st.subheader("Результат запроса")
                st.code(st.session_state.last_result)
        else:
            st.info("Загрузите CSV файл, чтобы увидеть данные.")

    # 2.2. Правая колонка
    with col_chat:
        st.header("Чат с данными")
        chat_container = st.container(height=500)
        for msg in st.session_state.messages:
            with chat_container.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if query := st.chat_input("Спросите что-нибудь...", disabled=not st.session_state.session_id):
            st.session_state.messages.append({"role": "user", "content": query})
            with chat_container.chat_message("user"):
                st.markdown(query)

            with st.spinner("Думаю..."):
                try:
                    df_info = df_code_analys.pd_getinfo(st.session_state.current_df)
                    system_prompt = prompts.prompt_code_generation(info=df_info, querry=query)
                    
                    response = st.session_state.client.chat.completions.parse(
                        model=st.session_state.model_name,
                        messages=system_prompt,
                        response_format=PandasCode,
                        temperature=0, top_p=0.95
                    )
                    parsed_result = response.choices[0].message.parsed
                    st.session_state.messages.append({"role": "assistant", "content": parsed_result.comment})
                    
                    exec_result = df_code_analys.normalize_and_execute_code(parsed_result.code, st.session_state.current_df)

                    if isinstance(exec_result, pd.DataFrame):
                        st.session_state.mgr.push_result(st.session_state.session_id, exec_result, code=parsed_result.code)
                        st.session_state.current_df = exec_result
                        st.session_state.last_result = None
                    else:
                        st.session_state.last_result = str(exec_result)
                    
                    st.rerun()

                except Exception as e:
                    error_message = f"Произошла ошибка: {e}"
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    st.error(error_message)

        if st.session_state.session_id:
            undo_col, redo_col = st.columns(2)
            if undo_col.button("Отменить (Undo)", use_container_width=True):
                with st.spinner("Отменяю..."):
                    st.session_state.mgr.undo(st.session_state.session_id)
                    st.session_state.current_df = st.session_state.mgr.load_current_df(st.session_state.session_id)
                    st.session_state.last_result = None
                    st.session_state.messages.append({"role": "assistant", "content": "Действие отменено."})
                    st.rerun()

            if redo_col.button("Повторить (Redo)", use_container_width=True):
                with st.spinner("Повторяю..."):
                    st.session_state.mgr.redo(st.session_state.session_id)
                    st.session_state.current_df = st.session_state.mgr.load_current_df(st.session_state.session_id)
                    st.session_state.last_result = None
                    st.session_state.messages.append({"role": "assistant", "content": "Действие повторено."})
                    st.rerun()
