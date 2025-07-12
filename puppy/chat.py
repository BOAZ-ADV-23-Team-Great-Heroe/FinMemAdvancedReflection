import os
import httpx
import logging
from typing import Dict, Any, Callable

class ChatOpenAICompatible:
    """
    OpenAI의 Chat Completion API와 호환되는 방식으로 통신하는 클래스.
    """
    def __init__(
        self,
        end_point: str,
        model: str,
        system_message: str,
        other_parameters: Dict[str, Any] = {},
    ):
        self.model = model
        self.end_point = end_point
        self.system_message = system_message

        self.other_parameters = other_parameters.copy()
        self.other_parameters.pop("system_message", None)
        
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            self.logger.setLevel(logging.ERROR)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            log_path = os.path.join("data", "04_model_output_log", "chat_errors.log")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            handler = logging.FileHandler(log_path, mode="a")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def end_point_func(self, prompt: str) -> str:
        """
        주어진 프롬프트를 사용하여 OpenAI 호환 API에 요청을 보내고 응답을 받습니다.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        }
        
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]

        data = {
            "model": self.model,
            "messages": messages, 
            **self.other_parameters,
        }

        try:
            with httpx.Client() as client:
                response = client.post(self.end_point, headers=headers, json=data, timeout=120)
                response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error occurred: {e.response.text}")
            raise e
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
            raise e
            
    def guardrail_endpoint(self) -> Callable:
        """
        guardrails 라이브러리와 함께 사용할 수 있는 엔드포인트 함수를 반환합니다.
        """
        def guardrail_func(prompt: str, **kwargs) -> str:
            return self.end_point_func(prompt)
        return guardrail_func