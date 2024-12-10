import os
import openai
from openai import OpenAIError
# Load your API key from an environment variable or .env file
openai.api_key = os.getenv('OPENAI_API_KEY')

try:
    # For text completion models like "text-davinci-003"
    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt="Hello, world!",
        max_tokens=5
    )
    print(response.choices[0].text.strip())

    # If using chat completion models like "gpt-3.5-turbo"
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[{"role": "user", "content": "Hello, world!"}],
    #     max_tokens=50
    # )
    # print(response.choices[0].message.content.strip())

except OpenAIError as e:
    print(f"API error: {e}")