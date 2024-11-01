# from openai import OpenAI
import openai
from config import API_KEY
from config import MODEL_NAME

openai.api_key = API_KEY



completion = openai.ChatCompletion.create(
    model = MODEL_NAME,
    messages = [
        {"role": "system", "content": "Just response the answer"},
        {"role": "user", "content": "What is 3+2"}
    ]
)

print(completion)