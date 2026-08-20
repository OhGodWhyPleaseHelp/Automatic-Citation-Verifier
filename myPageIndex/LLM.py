from openai import OpenAI, AsyncOpenAI
import time
import json
# from token_tracker import log_token_event
from utils import log_usage
import os
from dotenv import load_dotenv

load_dotenv()
MODEL_URL = os.environ.get("MODEL1_URL")
MODEL_NAME = os.environ.get("MODEL1_NAME")
MODEL_API_KEY = os.environ.get("MODEL1_API_KEY")

def sync_client():
    client = OpenAI(
        api_key=MODEL_API_KEY,
        base_url=MODEL_URL
    )
    MODEL = MODEL_NAME
    return client, MODEL


def async_client():
    client = AsyncOpenAI(
        api_key=MODEL_API_KEY,
        base_url=MODEL_URL
    )
    MODEL = MODEL_NAME
    return client, MODEL


def split_reply(model_reply: str, identifier: str) -> str:
    """
    The model reply usually comes with a <think> attached.
    This splits it and removes the identifier, like "```json".
    :param model_reply: output from the LLM.
    :param identifier: LLM identifier to show what type of output it is (e.g. python, json)
    :return: Only the output of the LLM
    """
    model_reply = model_reply.replace("Trying again...", '')
    split = model_reply.split(f"```{identifier}")
    if len(split) == 1 or '</think>' in split[-1]:
        if len(split) == 1:
            split = model_reply.split("</think>")
        else:
            split = split[-1].split("</think>")
        if len(split) == 1:
            raise Exception(f"Splitting the model reply has gone wrong!\n{model_reply}")
            # raise Exception(f"Splitting the model reply has gone wrong!")
    output = split[-1]
    output = output.replace("```", "")
    output = output.replace("```", "")

    json_content = output.replace('None', 'null')  # Replace Python None with JSON null
    json_content = json_content.replace('\n', ' ').replace('\r', ' ')  # Remove newlines
    json_content = ' '.join(json_content.split())  # Normalize whitespace
    try:
        json_output = json.loads(json_content)
        return json_output
    except Exception as e:
        # print(f"Could not JSONify model reply!\nModel Reply: {model_reply}\n{e}\nContinuing with this: {json_content}!")
        return json_content


def completion(client, model, messages, temperature):
    reply = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return reply


async def acompletion(client, model, messages, temperature):
    reply = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return reply


def llm_completion(prompt, prompt_name:str, chat_history=None, return_finish_reason=False):
    client, model = sync_client()
    max_retries = 10
    messages = list(chat_history) + [{"role": "user", "content": prompt}] if chat_history else [
        {"role": "user", "content": prompt}]
    content = ""
    for i in range(max_retries):
        try:
            response = completion(
                client=client,
                model=model,
                messages=messages,
                temperature=0,
            )
            content = response.choices[0].message.content
            content = split_reply(response.choices[0].message.content, 'json')

            log_usage(
                total_tokens=response.usage.total_tokens,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                prompt_name=prompt_name
            )

            if return_finish_reason:
                finish_reason = "max_output_reached" if response.choices[0].finish_reason == "length" else "finished"
                return content, finish_reason
            return content
        except Exception as e:
            print('************* Retrying *************')
            print(content)
            if i < max_retries - 1:
                time.sleep(1)
            else:
                print('Max retries reached for prompt: ' + prompt)
                if return_finish_reason:
                    return "", "error"
                return ""


async def llm_acompletion(client, model, prompt:str, prompt_name:str):
    max_retries = 10
    messages = [{"role": "user", "content": prompt}]
    for i in range(max_retries):
        try:
            response = await acompletion(
                client=client,
                model=model,
                messages=messages,
                temperature=0,
            )

            log_usage(
                total_tokens=response.usage.total_tokens,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                prompt_name=prompt_name
            )

            return split_reply(response.choices[0].message.content, 'json')
        except Exception as e:
            print('************* Retrying *************')
            # logging.error(f"Error: {e}")
            if i < max_retries - 1:
                await asyncio.sleep(1)
            else:
                logging.error('Max retries reached for prompt: ' + prompt)
                raise Exception(f"Max retrieves reached for prompt: {prompt}")
                return ""
