from openai import OpenAI
import pandas as pd
from pydantic import BaseModel
from api_integration import LLMClient
import df_code_analys



# Pydantic модель теперь проще, без eval_method
class PandasCode(BaseModel):
    code: str
    comment: str


import pandas as pd
import numpy as np

# Этот DataFrame будет использоваться и для генерации эталонных ответов,
# и как вход для LLM во время теста.
EVAL_DF = pd.DataFrame({
    'Survived': [0, 1, 1, 1, 0, 0],
    'Pclass': [3, 1, 3, 1, 3, 2],
    'Sex': ['male', 'female', 'female', 'female', 'male', 'male'],
    'Age': [22.0, 38.0, 26.0, 35.0, 35.0, 54.0],
    'Fare': [7.25, 71.28, 7.92, 53.1, 8.05, 26.0],
    'Embarked': ['S', 'C', 'S', 'S', 'S', 'S']
})
df = EVAL_DF
#df = pd.read_csv(r"C:\Users\tviva\Desktop\Titanic-Dataset.csv")
buffer = io.StringIO()
df.info(buf=buffer)
info = buffer.getvalue()
sample = df.sample(2).to_string()

# Промпт теперь не просит модель выбирать метод, а дает четкую инструкцию
# использовать `result` для возвращаемых значений.
prompt = [
    {
        "role": "system",
        "content": "Ты — программист-аналитик данных. Твоя задача — написать код на Python, чтобы ответить на запрос пользователя.\n"
                   "Для ответа на запросы пользователя используй только pandas"
                   "DataFrame для анализа доступен в переменной `df`.\n"
                   "Правила:\n"
                   "1. НИКОГДА не используй `print()` или `input()`.\n"
                   "2. Если запрос требует вернуть значение (число, строку, новый DataFrame), ОБЯЗАТЕЛЬНО присвой его переменной `result`.\n"
                   "3. Если запрос — это изменение `df` на месте (например, удаление столбцов), просто выполни код, не создавая `result`.\n"
                   "4. Не импортируй библиотеки.\n"
                   "5. В поле `comment` кратко опиши, что делает твой код."
    },
    {
        "role": "assistant",
        "content": f"{info}\n"
                   f"{sample}"
    },
    {
        "role": "user",
        "content": "Создай колонну, где возраст разбивается на категории, сохрани её в основном датасете"
    }
]
response = client.chat.completions.parse(
    model="qwen/qwen3-8b:free", #"hf.co/Vikhrmodels/QVikhr-3-4B-Instruction-GGUF:Q4_K_M",
    messages=prompt,
    response_format=PandasCode,  # Новый способ для pydantic
    temperature=0.3,
    top_p=0.95
)

# В новых версиях openai клиента результат нужно парсить из json
parsed_result = response.choices[0].message.parsed

print("--- Данные, полученные от LLM ---")
print(f"Комментарий: {parsed_result.comment}")

# ==============================================================================
# ШАГ 4: Заменяем старый блок выполнения новым, простым вызовом
# ==============================================================================

# Старый блок try/except полностью удален. Вместо него:
final_result = normalize_and_execute_code(parsed_result.code, df)

# Теперь в `final_result` находится либо вычисленное значение, либо измененный DataFrame.
# Вы можете продолжить работать с ним.

# Например, если вы хотите, чтобы исходный DataFrame `df` был обновлен
# результатами выполнения кода, можно добавить:
if isinstance(final_result, pd.DataFrame):
    df = final_result
    print("\n--- Основной DataFrame был успешно обновлен. ---")