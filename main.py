from openai import OpenAI
import pandas as pd
import io
import ast
from pydantic import BaseModel
from functions import df_code_analys,api_integration,prompts
from dotenv import load_dotenv
from endpoints import endpoints


load_dotenv() #загрузка переменных

client = api_integration.LLMClient(provider_name="openrouter")
df = pd.read_csv(r"C:\Users\tviva\Desktop\Titanic-Dataset.csv")
info = df_code_analys.pd_getinfo(df)
querry = "Сгруппируй выживших по полу"
system_prompt = prompts.prompt_code_generation(info=info,querry=querry)

response = client.chat.completions.parse(
  model="qwen/qwen3-8b:free",
  messages=system_prompt,
  response_format=endpoints.PandasCode,
  temperature=0,
  top_p=0.95
)

parsed_result = response.choices[0].message.parsed
print(f"Comment: {parsed_result.comment}")
print(f"Code: {parsed_result.code}")
print(df_code_analys.normalize_and_execute_code(parsed_result.code,df))
