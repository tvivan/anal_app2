import os
from openai import OpenAI
import pandas as pd
import numpy as np
from pydantic import BaseModel
from pandas.testing import assert_frame_equal, assert_series_equal
from functions import df_code_analys,api_integration,prompts
from dotenv import load_dotenv
# ВАЖНО: Убедитесь, что этот файл находится в той же папке,
# что и data_analyst.py, чтобы импорт сработал

load_dotenv()

class PandasCode(BaseModel):
    code: str
    comment: str
# --------------------------------------------------------------------------
# ШАГ 1: Определяем наш оценочный датасет (код из секции выше)
# --------------------------------------------------------------------------
EVAL_DF = pd.DataFrame({
    'Survived': [0, 1, 1, 1, 0, 0], 'Pclass': [3, 1, 3, 1, 3, 2],
    'Sex': ['male', 'female', 'female', 'female', 'male', 'male'],
    'Age': [22.0, 38.0, 26.0, 35.0, 35.0, 54.0],
    'Fare': [7.25, 71.28, 7.92, 53.1, 8.05, 26.0],
    'Embarked': ['S', 'C', 'S', 'S', 'S', 'S']
})

expected_df_test3 = EVAL_DF.copy()
expected_df_test3.drop(columns=['Embarked'], inplace=True)

expected_df_test4 = EVAL_DF.copy()
bins = [0, 18, 40, 100]
labels = ['Child', 'Adult', 'Senior']
expected_df_test4['AgeGroup'] = pd.cut(expected_df_test4['Age'], bins=bins, labels=labels)
expected_df_test4['AgeGroup'] = expected_df_test4['AgeGroup'].astype('category')  # Важно для сравнения

EVALUATION_DATASET = [
    {"id": "T01_SimpleMean", "prompt": "Каков средний возраст всех пассажиров?", "expected_result": 35.0},
    {"id": "T02_FilterAndCount", "prompt": "Сколько женщин выжило?", "expected_result": 3},
    {"id": "T03_DropColumn", "prompt": "Удали столбец Embarked из датасета.", "expected_result": expected_df_test3},
    {"id": "T04_CreateColumn",
     "prompt": "Создай новую колонку AgeGroup, разбив возраст на категории: 0-18 (Child), 18-40 (Adult), 40+ (Senior).",
     "expected_result": expected_df_test4},
    {"id": "T05_GroupBy", "prompt": "Сгруппируй по Pclass и посчитай среднюю стоимость билета Fare для каждого.",
     "expected_result": pd.Series([62.19, 26.00, 7.74], index=pd.Index([1, 2, 3], name='Pclass'), name='Fare')}
]

# --------------------------------------------------------------------------
# ШАГ 2: Функция для сравнения результатов
# --------------------------------------------------------------------------
def compare_results(actual, expected):
    """Сравнивает фактический результат с ожидаемым, обрабатывая разные типы."""
    try:
        if isinstance(expected, pd.DataFrame):
            # Используем специальную функцию pandas для точного сравнения
            assert_frame_equal(actual, expected, check_dtype=False)
        elif isinstance(expected, pd.Series):
            assert_series_equal(actual, expected, check_names=False, check_dtype=False, atol=0.01)
        elif isinstance(expected, (int, float)):
            assert np.isclose(actual, expected, atol=0.01)
        else:
            assert actual == expected
        return True, "OK"
    except AssertionError as e:
        return False, str(e)


# --------------------------------------------------------------------------
# ШАГ 3: Основной цикл тестирования
# --------------------------------------------------------------------------
def run_llm_evaluation():
    api_key = os.getenv("OPENROUTER_API_KEY")
    client = OpenAI(
        base_url='https://openrouter.ai/api/v1',
        api_key=api_key  # required, but unused
    )

    info = df_code_analys.pd_getinfo(EVAL_DF)

    results_summary = []

    print("=" * 20 + " ЗАПУСК ОЦЕНКИ LLM " + "=" * 20)

    for test_case in EVALUATION_DATASET:
        print(f"\n--- Тест: {test_case['id']} ---")
        print(f"Запрос: {test_case['prompt']}")

        # Готовим промпт для LLM
        system_prompt = [
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
                "content": f"{info}"
            },
            {
                "role": "user",
                "content": test_case['prompt']
            }
        ]

        try:
            response = client.chat.completions.parse(
                model="qwen/qwen3-8b:free",  # "hf.co/Vikhrmodels/QVikhr-3-4B-Instruction-GGUF:Q4_K_M",
                messages=system_prompt,
                response_format=PandasCode,  # Новый способ для pydantic
                temperature=0.3,
                top_p=0.95
            )
            parsed_result = response.choices[0].message.parsed

            # Выполняем код
            actual_result = df_code_analys.normalize_and_execute_code(parsed_result.code, EVAL_DF)

            # Сравниваем результаты
            is_pass, message = compare_results(actual_result, test_case['expected_result'])

            results_summary.append({
                "id": test_case['id'],
                "status": "PASS" if is_pass else "FAIL",
                "message": message
            })
            print(f"Статус: {'PASS' if is_pass else 'FAIL'}")
            if not is_pass:
                print(f"Причина: {message}")

        except Exception as e:
            print(f"Статус: ERROR")
            print(f"Причина: {e}")
            results_summary.append({
                "id": test_case['id'],
                "status": "ERROR",
                "message": str(e)
            })

    # --------------------------------------------------------------------------
    # ШАГ 4: Вывод итогового отчета
    # --------------------------------------------------------------------------
    print("\n\n" + "=" * 20 + " ИТОГОВЫЙ ОТЧЕТ " + "=" * 20)
    passed_count = sum(1 for r in results_summary if r['status'] == 'PASS')
    total_count = len(results_summary)

    print(f"Всего тестов: {total_count}")
    print(f"Успешно пройдено: {passed_count} ({passed_count / total_count:.0%})")

    if passed_count < total_count:
        print("\n--- Проваленные тесты ---")
        for result in results_summary:
            if result['status'] != 'PASS':
                print(f"- {result['id']}: {result['status']} ({result['message']})")


if __name__ == "__main__":
    run_llm_evaluation()