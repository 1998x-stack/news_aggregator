"""
DashScope LLM client for Alibaba Cloud's Qwen models
"""

import os
import json
from typing import Dict, Any, Optional, List
import requests
from loguru import logger


class DashScopeClient:
    """Client for DashScope API (Alibaba Cloud)"""

    def __init__(self, api_key: str = None, model: str = "qwen-max"):
        """Initialize DashScope client

        Args:
            api_key: DashScope API key (defaults to DASHSCOPE_API_KEY env var)
            model: Model to use (qwen-max, qwen-turbo, etc.)
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY must be set")

        self.model = model
        self.base_url = "https://dashscope.aliyuncs.com/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(f"DashScope client initialized with model: {model}")

    def generate(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        format: str = None,
    ) -> Dict[str, Any]:
        """Generate text using DashScope API

        Args:
            prompt: Input prompt
            model: Model to use (overrides default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            format: Response format (json, text, etc.)

        Returns:
            Dictionary with response and metadata
        """
        try:
            model_to_use = model or self.model

            # Prepare request body
            body = {
                "model": model_to_use,
                "input": {"messages": [{"role": "user", "content": prompt}]},
                "parameters": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "result_format": format or "text",
                },
            }

            logger.debug(f"Sending request to DashScope: {model_to_use}")

            response = requests.post(
                f"{self.base_url}/services/aigc/text-generation/generation",
                headers=self.headers,
                json=body,
                timeout=60,
            )

            response.raise_for_status()
            result = response.json()

            # Extract the generated text
            if "output" in result and "choices" in result["output"]:
                generated_text = result["output"]["choices"][0]["message"]["content"]

                return {
                    "response": generated_text,
                    "success": True,
                    "model": model_to_use,
                    "usage": result.get("usage", {}),
                }
            else:
                logger.error(f"Unexpected response format: {result}")
                return {
                    "response": "",
                    "success": False,
                    "error": "Unexpected response format",
                }

        except requests.exceptions.RequestException as e:
            logger.error(f"DashScope API request failed: {e}")
            return {"response": "", "success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error in DashScope client: {e}")
            return {"response": "", "success": False, "error": str(e)}

    def generate_json(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> Dict[str, Any]:
        """Generate JSON response using DashScope API

        Args:
            prompt: Input prompt (should specify JSON output)
            model: Model to use
            temperature: Sampling temperature (lower for JSON)
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with parsed JSON response
        """
        result = self.generate(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            format="json",
        )

        if result["success"]:
            try:
                # Try to parse the response as JSON
                response_text = result["response"].strip()

                # Remove markdown code blocks if present
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]

                response_text = response_text.strip()

                parsed_json = json.loads(response_text)
                result["parsed_json"] = parsed_json

            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse JSON response: {e}")
                result["parsed_json"] = None
                result["success"] = False
                result["error"] = f"JSON parse error: {e}"

        return result

    def batch_generate(
        self,
        prompts: List[str],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> List[Dict[str, Any]]:
        """Generate text for multiple prompts

        Args:
            prompts: List of input prompts
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response

        Returns:
            List of results
        """
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i + 1}/{len(prompts)}")
            result = self.generate(prompt, model, temperature, max_tokens)
            results.append(result)

        return results

    def check_health(self) -> bool:
        """Check if the DashScope API is accessible

        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Simple test - try to list available models
            response = requests.get(
                f"{self.base_url}/models", headers=self.headers, timeout=10
            )

            if response.status_code == 200:
                logger.info("DashScope API health check passed")
                return True
            else:
                logger.warning(
                    f"DashScope API health check failed: {response.status_code}"
                )
                return False

        except Exception as e:
            logger.error(f"DashScope API health check error: {e}")
            return False


# Global DashScope client instance
dashscope_client = None


def get_dashscope_client() -> DashScopeClient:
    """Get or create global DashScope client instance"""
    global dashscope_client

    if dashscope_client is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable not set")

        model = os.getenv("DASHSCOPE_MODEL", "qwen-max")
        dashscope_client = DashScopeClient(api_key=api_key, model=model)

    return dashscope_client
