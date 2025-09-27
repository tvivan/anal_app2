import os
import pandas as pd
from dotenv import load_dotenv
from functions import df_code_analys, api_integration, prompts
from endpoints import endpoints
from functions.memory import MemoryManager

#Задаем все условности
load_dotenv()
SESSION_ID = "default"
mgr = MemoryManager(redis_url=os.getenv("REDIS_URL"))
client = api_integration.LLMClient(provider_name="openrouter")

# Загружаем df в redis
meta = mgr.get_current_state_info(SESSION_ID)
if not meta:
    meta = mgr.init_session_from_csv(SESSION_ID, r"C:\Users\tviva\Desktop\Titanic-Dataset.csv")
    print("Исходный DataFrame загружен:", meta)
else:
    print("Найдено состояние:", meta)

df = mgr.load_current_df(SESSION_ID)

# Имитация диалога
print("\nВведи запрос для анализа данных.")
print("   Команды: 'undo', 'redo', 'exit'.\n")

while True:
    user_input = input("Запрос: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Завершение работы.")
        break

    elif user_input.lower() == "undo":
        meta = mgr.undo(SESSION_ID)
        if meta:
            df = mgr.load_current_df(SESSION_ID)
            print("Откат к предыдущему состоянию:", meta)
            print(df.head())
        else:
            print("Нет предыдущих состояний.")
        continue

    elif user_input.lower() == "redo":
        meta = mgr.redo(SESSION_ID)
        if meta:
            df = mgr.load_current_df(SESSION_ID)
            print("Перемотка вперёд:", meta)
            print(df.head())
        else:
            print("Нет следующих состояний.")
        continue

    # получаем info о текущем df
    info = df_code_analys.pd_getinfo(df)

    # формируем промпт
    system_prompt = prompts.prompt_code_generation(info=info, querry=user_input)

    print("Отправляем запрос модели...")
    response = client.chat.completions.parse(
        model="qwen/qwen3-30b-a3b:free",
        messages=system_prompt,
        response_format=endpoints.PandasCode,
        temperature=0,
        top_p=0.95
    )

    parsed_result = response.choices[0].message.parsed

    # выводим результат от модели
    print("\nКомментарий модели:", parsed_result.comment)
    print("Сгенерированный код:\n", parsed_result.code)

    # выполняем код
    execution_result = df_code_analys.normalize_and_execute_code(parsed_result.code, df)

    if isinstance(execution_result, pd.DataFrame):
        meta = mgr.push_result(SESSION_ID, execution_result, code=parsed_result.code)
        df = execution_result
        print("DataFrame изменён. Метаданные:", meta)
        print(df.head())
    else:
        print("Результат выполнения запроса:", execution_result)
