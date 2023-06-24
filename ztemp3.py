#%%
import re, time, json, os

import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

#%%

def queryGPT(query, system_query = None, model = "gpt-3.5-turbo"):
    # model = "gpt-4"

    output_raw = None
    messages = [{"role": "user", "content": query},]
    if system_query:
        messages.insert(0, {"role": "system", "content": F"{system_query}"})

    print(messages)
    try:
        output_raw = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
            n=1
            )
        if output_raw != None:
            print(F"Received response")

    except (openai.error.Timeout,
            openai.error.APIError,
            openai.error.APIConnectionError,
            openai.error.RateLimitError) as e:
        print(f"OpenAI API error: {e}\nRetrying...")
        time.sleep(1.5)
        output_raw = queryGPT(query=query, system_query=system_query, model=model)
        
    return output_raw