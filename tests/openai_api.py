from openai import OpenAI
import pandas as pd
import io
from pydantic import BaseModel
from typing import Literal

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama' # required, but unused
)
class Pandas_code(BaseModel):
    code: str
    eval_method: Literal["eval", "exec"]
    comment: str


df = pd.read_csv(r"C:\Users\tviva\Desktop\Titanic-Dataset.csv")
buffer = io.StringIO()
df.info(buf=buffer)
info_string = buffer.getvalue()
sample = df.sample(5).to_string()
prompt = [
    {
        "role": "system",
        "content": "Ты — программист-аналитик данных. Твоя задача — написать код на Python, чтобы ответить на запрос пользователя. \n"
                   "Тебе нужно думать как аналитик данных, аналитикам всегда нужны подробные цифры и статистика, старайся делать код в этом направлении \n"
                   "Не отвечай ничем, кроме самого кода. Никогда не используй print для вывода или input для ввода\n"
                   "Для вычислений используй только код, можешь написать несколько строчек, елси нужно.\n"
                   "DataFrame содержится в переменной df. \n"
                   "Не импортируй никакие библиотеки, считай, что они уже импортированы. \n"
                   "Определи правильно eval или exec. Eval если пользователю нужен ответ числовой, а exec если он хочет сделать изменение в df \n"
                   "В comment дай обоснование своего ответа"
    },
    {
        "role": "assistant",
        "content": create_dataset_context(df)
    },
    {
        "role": "user",
        "content": "Перекодируй датасет так, чтобы можно было использовать его для машинного обучения, лишние колонны можешь удалить"
    }
]
response = client.chat.completions.parse(
  model="qwen/qwen3-8b:free",
  messages=prompt,
  response_format=Pandas_code,
  temperature=0,
  top_p=0.95
)

parsed_result = response.choices[0].message.parsed
print(parsed_result)
# Теперь parsed_result — это экземпляр вашего класса Pandas_code
# и вы можете обращаться к его полям напрямую
code_string = parsed_result.code
method = parsed_result.eval_method

print("--- Извлеченные данные ---")
print(f"Код для выполнения: {code_string}")
print(f"Метод: {method}")

try:
    if method == 'eval':
        # Разделяем код на строки
        code_lines = code_string.strip().split('\n')

        # Все строки, кроме последней - это подготовительный код
        setup_code = '\n'.join(code_lines[:-1])

        # Последняя строка - это выражение, которое нужно вычислить
        result_expression = code_lines[-1]

        # Создаем локальное окружение для выполнения кода
        local_scope = {}

        # Выполняем подготовительный код с помощью exec в этом окружении
        # Передаем df через globals, чтобы он был доступен
        exec(setup_code, {'df': df}, local_scope)

        # Вычисляем итоговое выражение с помощью eval, используя то же окружение
        result = eval(result_expression, {'df': df}, local_scope)

        print(f"\n--- Результат выполнения (eval) --- \n{result}")

    elif method == 'exec':
        # Для exec ничего менять не нужно, он и так умеет работать с многострочным кодом
        exec(code_string, {'df': df})
        print("\n--- Код выполнен (exec), DataFrame мог быть изменен ---")
        print(df.head())

except Exception as e:
    print(f"\n--- Ошибка при выполнении кода: {e} ---")