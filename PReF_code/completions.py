import pdb
from openai import OpenAI, RateLimitError
import os
import time
# vvv 
OPENAI_API_KEY = "PLEASE FILL IN"
client = OpenAI(
    api_key = OPENAI_API_KEY
)

def get_completion(prompt, system_prompt='You are a helpful instruction-following assistant.', temp=0.0, model='gpt-4o', messages=None, return_messages=False):
    try:
        if messages is None:
            if system_prompt is None:
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
        else:
            messages.append({'role': 'user', 'content': prompt})

        response = client.chat.completions.create(
            model=model,
            messages = messages,
            temperature=temp
        )

        message = response.choices[0].message
        response_text = message.content
        messages.append({'role': message.role, 'content': message.content})
        if return_messages:
            return messages
        else:
            return response_text
    except RateLimitError as e:
        print(f"Rate limit hit, waiting 60 seconds... {str(e)}")
        time.sleep(60)
        return get_completion(prompt, system_prompt, temp, model, messages, return_messages)
    except Exception as e:
        print(f"Error in get_completion: {str(e)}")
        raise  # Re-raise the exception to be caught by mt_run

if __name__ == '__main__':

    system_prompt = "You are a helpful assistant."
    prompt = "Hello"

    get_completion(prompt, system_prompt)