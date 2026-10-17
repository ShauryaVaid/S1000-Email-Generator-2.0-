"""AI Client for Gemini and Ollama API interactions with token control and error resilience."""

import requests
import time
import json
import re
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


class AIClient:
    """Unified AI client supporting Google Gemini and Ollama providers."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AI client.

        Args:
            config: Configuration dictionary with provider, model, base_url, api_key, etc.
        """
        self.provider = config.get('provider', 'gemini')
        self.model = config.get('model', 'gemini-2.5-flash')
        self.embedding_model = config.get('embedding_model', 'gemini-embedding-001')
        self.base_url = config.get('base_url', '')
        self.api_key = config.get('api_key', '')
        self.delay = config.get('delay', 0.05)
        self.max_workers = config.get('max_workers', 20)
        self.max_output_tokens = config.get('max_output_tokens', 800)
        self.temperature = config.get('temperature', 0.5)
        self._ollama_embedding_endpoint: Optional[str] = None

    def get_embedding(self, text: str) -> List[float]:
        """Generate single embedding vector."""
        result = self.get_embeddings_batch([text])
        return result[0] if result else []

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings using threading."""
        if self.provider == "gemini":
            return self._gemini_parallel(texts)
        elif self.provider in ["ollama", "ollama-local"]:
            return self._ollama_parallel(texts)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text completion from LLM with token control and retry logic."""
        if self.provider == "gemini":
            return self._gemini_generate(prompt, system_prompt)
        elif self.provider in ["ollama", "ollama-local"]:
            return self._ollama_generate(prompt, system_prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        retry_count: int = 3
    ) -> Dict[str, Any]:
        """
        Generate JSON response from LLM with automatic parsing and retry logic.

        Args:
            prompt: User prompt
            system_prompt: System instructions
            retry_count: Number of retry attempts if JSON parsing fails

        Returns:
            Parsed JSON dictionary
        """
        for attempt in range(retry_count):
            try:
                json_prompt = (
                    prompt +
                    "\n\nYou MUST respond with valid JSON only. "
                    "No explanations, no markdown code blocks, just raw JSON."
                )

                response = self.generate_text(json_prompt, system_prompt)
                parsed_json = self._extract_json(response)

                if parsed_json:
                    return parsed_json
                else:
                    if attempt < retry_count - 1:
                        print(f"Warning: JSON parsing failed, retrying ({attempt + 1}/{retry_count})...")
                        time.sleep(1)
                    else:
                        raise ValueError("Failed to parse JSON after multiple attempts")

            except Exception as e:
                if attempt < retry_count - 1:
                    print(f"Error generating JSON: {e}, retrying...")
                    time.sleep(1)
                else:
                    raise

        return {}

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from text (handles markdown code blocks)."""
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return None

    # ========== GEMINI METHODS (Cloud) ==========

    def _gemini_generate(self, prompt: str, system_prompt: Optional[str] = None, max_retries: int = 2) -> str:
        """
        Generate text using Google Gemini API with token limit controls and rate-limit backoff.
        """
        if not self.api_key:
            raise ValueError(
                "Gemini API key is not configured. Please set GEMINI_API_KEY in Streamlit secrets or environment variables."
            )

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.model}:generateContent?key={self.api_key}"
        )

        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
                "topP": 0.95
            }
        }

        if system_prompt:
            payload["system_instruction"] = {
                "parts": [{"text": system_prompt}]
            }

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(url, json=payload, timeout=60)

                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get('candidates', [])
                    if candidates and 'content' in candidates[0]:
                        parts = candidates[0]['content'].get('parts', [])
                        if parts and 'text' in parts[0]:
                            return parts[0]['text'].strip()
                    return "Error: Empty response returned from Gemini API."

                elif response.status_code == 429:
                    last_error = "Rate limit reached (429). Waiting before retrying..."
                    if attempt < max_retries:
                        time.sleep(2 * (attempt + 1))
                        continue
                    return "⚠️ API Rate limit reached. Please wait a moment before sending another request."

                elif response.status_code == 404:
                    err_msg = response.json().get('error', {}).get('message', 'Model not found')
                    return f"⚠️ Model '{self.model}' unavailable: {err_msg}. Please select another model in settings."

                elif response.status_code == 400:
                    err_msg = response.json().get('error', {}).get('message', 'Bad request')
                    return f"⚠️ API Error (400): {err_msg}"

                else:
                    response.raise_for_status()

            except requests.exceptions.RequestException as e:
                last_error = str(e)
                if attempt < max_retries:
                    time.sleep(1.5)
                    continue

        return f"⚠️ Network/API Error: {last_error}"

    def _gemini_parallel(self, texts: List[str]) -> List[List[float]]:
        """Gemini parallel embedding generation using threading."""
        if not self.api_key:
            return [[] for _ in texts]

        embed_model = self.embedding_model or 'gemini-embedding-001'

        def embed_single(text: str) -> List[float]:
            try:
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/"
                    f"models/{embed_model}:embedContent?key={self.api_key}"
                )
                payload = {"content": {"parts": [{"text": text}]}}
                response = requests.post(url, json=payload, timeout=30)
                if response.status_code == 200:
                    return response.json().get('embedding', {}).get('values', [])
                return []
            except Exception as e:
                print(f"Gemini embedding error: {e}")
                return []

        results = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=min(self.max_workers, 10)) as executor:
            future_to_idx = {
                executor.submit(embed_single, text): idx
                for idx, text in enumerate(texts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return results

    # ========== OLLAMA METHODS (Local) ==========

    def _ollama_generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Ollama native REST API."""
        base_url = self.base_url or "http://localhost:11434"
        url = f"{base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_output_tokens
            }
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json().get('response', '').strip()
        except requests.exceptions.ConnectionError:
            return "⚠️ Could not connect to local Ollama server at http://localhost:11434. Make sure 'ollama serve' is running, or switch to Google Gemini in the sidebar."
        except Exception as e:
            return f"⚠️ Ollama Error: {str(e)}"

    def _ollama_parallel(self, texts: List[str]) -> List[List[float]]:
        """Ollama parallel embedding generation using threading."""
        base_url = self.base_url or "http://localhost:11434"

        def embed_single(text: str) -> List[float]:
            if self._ollama_embedding_endpoint:
                try:
                    payload = {"model": self.embedding_model or self.model, "prompt": text}
                    response = requests.post(self._ollama_embedding_endpoint, json=payload, timeout=30)
                    response.raise_for_status()
                    result = response.json()
                    return result.get('embedding') or result.get('embeddings', [[]])[0]
                except Exception:
                    pass

            endpoints = [
                f"{base_url}/api/embed",
                f"{base_url}/api/embeddings"
            ]

            for endpoint in endpoints:
                try:
                    payload = {"model": self.embedding_model or self.model, "prompt": text}
                    response = requests.post(endpoint, json=payload, timeout=30)
                    if response.status_code in [404, 501]:
                        continue
                    response.raise_for_status()
                    result = response.json()
                    if 'embedding' in result or 'embeddings' in result:
                        self._ollama_embedding_endpoint = endpoint
                        return result.get('embedding') or result.get('embeddings', [[]])[0]
                except Exception:
                    continue
            return []

        results = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(embed_single, text): idx
                for idx, text in enumerate(texts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return results
