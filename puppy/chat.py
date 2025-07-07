# import os
# import httpx
# import json
# import subprocess
# import logging
# from abc import ABC
# from typing import Callable, Union, Dict, Any, Union

# ### when use tgi model
# api_key = '-' 

# def build_llama2_prompt(messages):
#     startPrompt = "<s>[INST] "
#     endPrompt = " [/INST]"
#     conversation = []
#     for index, message in enumerate(messages):
#         if message["role"] == "system" and index == 0:
#             conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")
#         elif message["role"] == "user":
#             conversation.append(message["content"].strip())
#         else:
#             conversation.append(f" [/INST] {message['content'].strip()}</s><s>[INST] ")

#     return startPrompt + "".join(conversation) + endPrompt


# class LongerThanContextError(Exception):
#     pass

# class ChatOpenAICompatible(ABC):
#     def __init__(
#         self,
#         end_point: str,
#         model="gemini-pro",
#         system_message: str = "You are a helpful assistant.",
#         other_parameters: Union[Dict[str, Any], None] = None,
#     ):
#         api_key = os.environ.get("OPENAI_API_KEY", "-")
#         self.end_point = end_point
#         self.model = model
#         self.system_message = system_message
        
#         # 로깅 추가
#         self.logger = logging.getLogger(__name__)
#         self.logger.setLevel(logging.INFO)
#         self.logger.info(f"Initializing ChatOpenAICompatible with model: {model}")
#         self.logger.info(f"API Key exists: {bool(api_key)}")
#         self.logger.info(f"API Key prefix: {api_key[:8]}...")
        
#         if self.model.startswith("gemini-pro"):
#             proc_result = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True)
#             access_token = proc_result.stdout.strip()
#             self.headers = {     "Authorization": f"Bearer {access_token}",
#                                 "Content-Type": "application/json",
#                             }
#         elif self.model.startswith("tgi"):
#             self.headers = {
#                         'Content-Type': 'application/json'
#                     }   
#         else:
#             self.headers = {
#                 "Authorization": f"Bearer {api_key}",
#                 "Content-Type": "application/json",
#             }
#             self.other_parameters = {} if other_parameters is None else other_parameters
#             self.logger.info(f"Headers: {self.headers}")
#             self.logger.info(f"Other parameters: {self.other_parameters}")

#     def parse_response(self, response: httpx.Response) -> str:
#         if self.model.startswith("gpt"):
#             response_out = response.json()
#             return response_out["choices"][0]["message"]["content"]
#         elif self.model.startswith("gemini-pro"):
#             response_out = response.json()
#             return response_out["candidates"][0]["content"]["parts"][0]["text"]
#         elif self.model.startswith("tgi"):
#             response_out = response.json()#[0]
#             return response_out["generated_text"]
#         else:
#             raise NotImplementedError(f"Model {self.model} not implemented")

#     def guardrail_endpoint(self) -> Callable:
#         def end_point(input: str, **kwargs) -> str:
#             self.logger.info(f"Guardrail endpoint called with input: {input[:100]}...")
#             input_str = [
#                     {"role": "system", "content": "You are a helpful assistant only capable of communicating with valid JSON, and no other text."},
#                     {"role": "user", "content": f"{input}"},
#                 ]
            
#             if self.model.startswith("gemini-pro"):
#                 input_prompts = {"role": "USER",
#                                 "parts": { "text": input_str[1]["content"]}
#                                     }
#                 payload = {"contents": input_prompts,
#                             "generation_config": {
#                                                 "temperature": 0.2,
#                                                 "top_p": 0.1,
#                                                 "top_k": 16,
#                                                 "max_output_tokens": 2048,
#                                                 "candidate_count": 1,
#                                                 "stop_sequences": []
#                                                 },
#                             "safety_settings": {
#                                                 "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
#                                                 "threshold": "BLOCK_LOW_AND_ABOVE"
#                                                 }
#                         }
#                 self.logger.info(f"Sending request to Gemini API with payload: {payload}")
#                 response = httpx.post(url = self.end_point, headers= self.headers, json=payload, timeout=600.0 )
                
#             elif self.model.startswith("tgi"):
#                 llama_input_str = build_llama2_prompt(input_str)
#                 payload = {
#                 "inputs": llama_input_str,
#                 "parameters": {
#                                 "do_sample": True,
#                                 "top_p": 0.6,
#                                 "temperature": 0.8,
#                                 "top_k": 50,
#                                 "max_new_tokens": 256,
#                                 "repetition_penalty": 1.03,
#                                 "stop": ["</s>"]
#                             }
#                             }
#                 self.logger.info(f"Sending request to TGI API with payload: {payload}")
#                 response = httpx.post(
#                     self.end_point, headers=self.headers, json=payload, timeout=600.0
#                 )
#             else:
#                 payload = {
#                     "model": self.model,
#                     "messages": input_str,
#                 }
#                 payload.update(self.other_parameters)
#                 self.logger.info(f"Sending request to OpenAI API with payload: {payload}")
#                 response = httpx.post(
#                     self.end_point, headers=self.headers, json=payload, timeout=600.0
#                 )
#             try:
#                 response.raise_for_status()
#                 self.logger.info(f"API response status: {response.status_code}")
#                 self.logger.info(f"API response: {response.text[:200]}...")
#             except httpx.HTTPStatusError as e:
#                 self.logger.error(f"HTTP error occurred: {str(e)}")
#                 self.logger.error(f"Response text: {response.text}")
#                 if (response.status_code == 422) and ("must have less than" in response.text):
#                     raise LongerThanContextError
#                 else:
#                     raise e

#             return self.parse_response(response)

#         return end_point

# chat.py (Final Version with Compatibility Fix)

import os
import httpx
import json
import subprocess
import logging
from abc import ABC
from typing import Callable, Union, Dict, Any

class LongerThanContextError(Exception):
    pass

def build_llama2_prompt(messages):
    startPrompt = "<s>[INST] "
    endPrompt = " [/INST]"
    conversation = []
    for index, message in enumerate(messages):
        if message["role"] == "system" and index == 0:
            conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")
        elif message["role"] == "user":
            conversation.append(message["content"].strip())
        else:
            conversation.append(f" [/INST] {message['content'].strip()}</s><s>[INST] ")
    return startPrompt + "".join(conversation) + endPrompt


class ChatOpenAICompatible(ABC):
    def __init__(
        self,
        end_point: str,
        model="gemini-pro",
        system_message: str = "You are a helpful assistant.",
        other_parameters: Union[Dict[str, Any], None] = None,
    ):
        api_key = os.environ.get("OPENAI_API_KEY", "-")
        self.end_point = end_point
        self.model = model
        self.system_message = system_message
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing ChatOpenAICompatible with model: {model}")

        if self.model.startswith("gemini-pro"):
            proc_result = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True)
            access_token = proc_result.stdout.strip()
            self.headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }
        elif self.model.startswith("tgi"):
            self.headers = {'Content-Type': 'application/json'}   
        else:
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        self.other_parameters = {} if other_parameters is None else other_parameters

    def parse_response(self, response: httpx.Response) -> str:
        if self.model.startswith("gpt"):
            response_out = response.json()
            # +++ 추가된 부분: 오류 응답 처리 로직 +++
            if "error" in response_out:
                self.logger.error(f"OpenAI API Error: {response_out['error']}")
                # guardrails가 이해할 수 있는 JSON 오류 형태로 반환
                return json.dumps({"error": response_out['error']['message']})
            return response_out["choices"][0]["message"]["content"]
        
        elif self.model.startswith("gemini-pro"):
            response_out = response.json()
            return response_out["candidates"][0]["content"]["parts"][0]["text"]
        elif self.model.startswith("tgi"):
            response_out = response.json()
            return response_out["generated_text"]
        else:
            raise NotImplementedError(f"Model {self.model} not implemented")

    def guardrail_endpoint(self) -> Callable:
        """
        +++ 수정된 부분: guardrails와의 호환성을 위해 **kwargs를 받아 처리합니다. +++
        이제 'You must provide messages' 오류가 해결됩니다.
        """
        def end_point(prompt: str, **kwargs) -> str:
            self.logger.info(f"Guardrail endpoint called. kwargs: {kwargs}")

            # guardrails는 'messages' 인자를 직접 전달할 수 있습니다.
            # 이 인자가 있는지 확인하여 payload를 구성합니다.
            messages_from_kwargs = kwargs.get("messages")

            if messages_from_kwargs:
                # guardrails가 제공하는 messages를 우선적으로 사용
                input_str = messages_from_kwargs
            else:
                # 기존 방식: 시스템 메시지와 사용자 프롬프트 조합
                input_str = [
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt},
                ]
            
            # --- 이후 API 호출 로직은 이전과 동일 ---

            if self.model.startswith("gemini-pro"):
                # Gemini 호출 로직
                # ... (이전과 동일)
                pass
                
            elif self.model.startswith("tgi"):
                # TGI 호출 로직
                # ... (이전과 동일)
                pass

            else: # Default to OpenAI GPT model
                payload = {
                    "model": self.model,
                    "messages": input_str,
                }
                payload.update(self.other_parameters)
                self.logger.info(f"Sending request to OpenAI API with payload: {json.dumps(payload, indent=2)}")
                
                try:
                    response = httpx.post(
                        self.end_point, headers=self.headers, json=payload, timeout=600.0
                    )
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    self.logger.error(f"HTTP error occurred: {e.response.text}")
                    if (e.response.status_code == 422) and ("must have less than" in e.response.text):
                        raise LongerThanContextError from e
                    raise e
                except Exception as e:
                    self.logger.error(f"An unexpected error occurred: {e}")
                    raise e

            return self.parse_response(response)

        return end_point