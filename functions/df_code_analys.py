import pandas as pd
import io
import ast
#df = pd.read_csv(r"C:\Users\tviva\Desktop\Titanic-Dataset.csv")
def pd_getinfo(df: pd.DataFrame):
    """
    Function for briefly data analys
    :param df: current df user working with
    :return: info_string: df.info()
             sample: df.sample(5)
    """
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_string = buffer.getvalue()
    sample = df.sample(5).to_string()
    result = info_string + sample
    return result
#print(pd_getinfo(df))
def normalize_and_execute_code(code_string: str, dataframe: pd.DataFrame):
    """
    Code normalization by adding `result = ...` to one string
    and compile as exec.
    :param code_string: evaluted code by LLM
    :param dataframe: current df user working with
    :return: result or df depending on what code was generated
    """
    print(f"--- Исходный код от LLM ---\n{code_string}\n")

    corrected_code = code_string.strip()
    try:
        tree = ast.parse(corrected_code)
        if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
            print(">>> Анализ: Код является одиночным выражением. Нормализуем для захвата результата.")
            corrected_code = f"result = {corrected_code}"
        else:
            print(">>> Анализ: Код содержит инструкции или уже присваивает результат.")
    except SyntaxError:
        print(">>> Анализ: Не удалось разобрать код, будет выполнен как есть.")

    print(f"\n--- Код для выполнения ---\n{corrected_code}\n")

    try:
        df_copy = dataframe.copy()
        local_scope = {'df': df_copy, 'pd': pd}
        exec(corrected_code, {}, local_scope)

        if 'result' in local_scope:
            result = local_scope['result']
            print(f"--- Результат выполнения (значение 'result' найдено) ---")
            if result is None:
                print("Результат - None. Вероятно, DataFrame был изменен на месте.")
                print("Новое состояние DataFrame:")
                print(df_copy.head())
                return df_copy
            else:
                print("Полученное значение:")
                print(result)
                return result
        else:
            print("--- Код выполнен, DataFrame был изменен ('result' не найден) ---")
            print("Новое состояние DataFrame:")
            print(df_copy.head())
            return df_copy
    except Exception as e:
        print(f"\n--- Ошибка при выполнении кода: {e} ---")
        return None

