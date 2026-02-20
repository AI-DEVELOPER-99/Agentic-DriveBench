import os
import re
import time
import functools
import json
import random
import requests


def retry_on_failure(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        while True:
            try:
                reply, total_tokens = func(*args, **kwargs)
                return reply, total_tokens
            except Exception as e:
                print(f"An error occurred: {e}. Retrying in 5 seconds...")
                time.sleep(5)
    return wrapper


class GPTEvaluation:
    def __init__(self, 
                 api_key=None, 
                 temperature=0.0,
                 max_tokens=3000,
                 log_file=None,
                 use_local=False,
                 ollama_model: str = "gpt-oss:20b",
                 ollama_url: str = "http://localhost:11434"):
        """GPT evaluation API.

        The evaluation can either hit the OpenAI cloud (default) or a
        local Ollama model.  If ``use_local`` is True the ``api_key`` is
        ignored and ``ollama_model``/``ollama_url`` are used instead.
        """
        
        self.api_key = api_key
        self.model = "gpt-3.5-turbo"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.content = []
        self.log_file = log_file
        # parameters for local evaluation
        self.use_local = use_local
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url.rstrip('/')
        
        # `log_data` should always be a dictionary mapping the unique key
        # to previously computed scores.  Previously it defaulted to an
        # empty list if no log file existed, which caused the
        # AttributeError during `.get(...)` calls.  Initialize it as an
        # empty dict and only replace it when resuming from disk.
        self.log_data = {}
        
        # If a log file already exists, load its entries so we can skip
        # repeated GPT calls when resuming.
        if self.log_file is not None:
            log_dir = os.path.dirname(self.log_file)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            if os.path.exists(self.log_file):
                self.resume_evaluation()

    def resume_evaluation(self):
        """Resumes evaluation from the last checkpoint in the log file.
        """
        if self.log_file is None:
            return
        log_data = []
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    log_data.append(json.loads(line))
                except:
                    pass
        self.log_data = {}
        for entry in log_data:
            key = f"{entry['scene_token']}_{entry['frame_token']}_{entry['question']}"
            self.log_data[key] = entry["gpt_score"]

    def addTextPrompt(self, textPrompt: str):
        self.content.append({
            "type": "text",
            "text": textPrompt
        })

    @retry_on_failure
    def request_chatgpt(self):
        # depending on configuration, either call OpenAI cloud or local
        # Ollama model.
        if self.use_local:
            # local call via Ollama HTTP API
            payload = {
                "model": self.ollama_model,
                "prompt": "\n".join([msg["text"] for msg in self.content]),
                "stream": False,
                "options": {"temperature": self.temperature}
            }
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload)
            response.raise_for_status()
            resp_json = response.json()
            reply = resp_json.get('response', '')
            # Ollama doesn't return token counts in its standard JSON
            total_tokens = resp_json.get('usage', {}).get('total_tokens', 0)
            return reply, total_tokens
        else:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload = {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": "\n".join([msg["text"] for msg in self.content])
                    }
                ],
                "max_tokens": self.max_tokens,
            }
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            if response.status_code != 200:
                raise Exception(f"Request failed with status code {response.status_code}: {response.text}")
            response_json = response.json()
            reply = response_json['choices'][0]['message']['content']
            total_tokens = response_json['usage']['total_tokens']
            return reply, total_tokens

    def prepare_chatgpt_message(self, prompt):
        system_message = "You are an evaluator who rates answers based on their closeness to the correct answer."
        self.addTextPrompt(system_message)
        self.addTextPrompt(prompt)

    def call_chatgpt(self, prompt):
        self.content = []  # Reset for each call.
        self.prepare_chatgpt_message(prompt)
        reply, total_tokens = self.request_chatgpt()
        return reply, total_tokens

    def extract_answer(self, reply):
        """Extracts an integer score from a GPT reply.

        The cloud and local models may format their responses differently
        (e.g. markdown bold, bullet lists, etc.).  To be resilient we first
        strip common markdown characters and then look for the numeric score.
        """
        # remove asterisks/underscore that may be added for bold/italic markup
        cleaned = re.sub(r"[\*_]", "", reply)
        # collapse newlines so the regex can match across lines
        cleaned = cleaned.replace("\n", " ")

        pattern = r"Total Score[:\s]*([0-9]{1,3})\b"
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"No valid score found in the reply:\n{reply}")

    def log(self, input_tuple):
        """
        Logs the provided prompt, GPT output, final score, and input data.
        The log entry automatically includes extra key-value pairs from extra_log_info.
        """
        if self.log_file is None:
            return
        with open(self.log_file, 'a') as f:
            json.dump(input_tuple, f, separators=(',', ':'))
            f.write('\n')

    def forward(self, data):
        """
        Expects data as a tuple: ((question, predicted, GT, desc), prompt_template).
        It formats the prompt accordingly, calls GPT, extracts the score,
        logs the result (including an extra 'input_data' field), and returns the score.
        """
        (input_tuple, prompt_template) = data
        scene_token = input_tuple['scene_token']
        frame_token = input_tuple['frame_token']
        question = input_tuple['question']
        answer = input_tuple['answer']
        pred = input_tuple['pred']
        desc = input_tuple['desc']

        success = False
        reply = None

        # skip questions if in log
        if self.log_data is not None:
            key = f"{scene_token}_{frame_token}_{question}"
            reply = self.log_data.get(key, None)
            success = True if reply is not None else False

        if not desc:
            prompt = prompt_template.format(GT=answer, PRED=pred, QUESTION=question)
        else:
            prompt = prompt_template.format(GT=answer, PRED=pred, QUESTION=question, DESC=desc)
            
        while not success:
            try:
                reply, total_tokens = self.call_chatgpt(prompt)
                success = True
            except Exception as e:
                print(f"Request failed: {e}. Retrying in 5 seconds...")
                time.sleep(5)
        if isinstance(reply, str):
            reply = reply.strip()
        final_score = self.extract_answer(reply)

        input_tuple['gpt_score'] = reply
        self.log(input_tuple)
        
        return final_score