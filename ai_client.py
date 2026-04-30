"""AI Client for Ollama and Gemini API interactions."""

import requests
import time
import json
import re
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


class AIClient:
    """Unified AI client supporting Ollama and Gemini providers."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AI client.

        Args:
            config: Configuration dictionary with provider, model, base_url, api_key, etc.
        """
        self.provider = config['provider']
        self.model = config['model']
        self.base_url = config.get('base_url', '')
        self.api_key = config.get('api_key', '')
        self.delay = config.get('delay', 0.05)
        self.max_workers = config.get('max_workers', 50)
        self._ollama_embedding_endpoint: Optional[str] = None

    def get_embedding(self, text: str) -> List[float]:
        """Generate single embedding vector."""
        result = self.get_embeddings_batch([text])
        return result[0] if result else []

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings using threading."""
        if self.provider in ["ollama", "ollama-cloud", "minimax-cloud"]:
            return self._ollama_parallel(texts)
        elif self.provider == "gemini":
            return self._gemini_parallel(texts)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text completion from LLM."""
        if self.provider == "minimax-cloud":
            return self._minimax_cloud_generate(prompt, system_prompt)
        elif self.provider in ["ollama", "ollama-cloud"]:
            return self._ollama_generate(prompt, system_prompt)
        elif self.provider == "gemini":
            return self._gemini_generate(prompt, system_prompt)
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
                        print(f"   Warning: JSON parsing failed, retrying ({attempt + 1}/{retry_count})...")
                        time.sleep(1)
                    else:
                        raise ValueError("Failed to parse JSON after multiple attempts")

            except Exception as e:
                if attempt < retry_count - 1:
                    print(f"   Error generating JSON: {e}, retrying...")
                    time.sleep(1)
                else:
                    raise

        return {}

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from text (handles markdown code blocks)."""
        # Try direct JSON parse first
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try to extract from markdown code block
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find any JSON object in the text
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return None

    # ========== OLLAMA METHODS ==========

    def _ollama_generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Ollama native REST API."""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        if system_prompt:
            payload["system"] = system_prompt

        headers = {}
        # Add Bearer token auth for Ollama Cloud / Minimax Cloud
        if self.provider in ["ollama-cloud", "minimax-cloud"] and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            print(f"Error generating text: {e}")
            return ""

    def _ollama_parallel(self, texts: List[str]) -> List[List[float]]:
        """Ollama parallel embedding generation using threading."""
        # Prepare auth headers for Ollama Cloud / Minimax Cloud
        headers = {}
        if self.provider in ["ollama-cloud", "minimax-cloud"] and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        def embed_single(text: str) -> List[float]:
            # Check cache first
            if self._ollama_embedding_endpoint:
                try:
                    if "embeddings" in self._ollama_embedding_endpoint:
                        payload = {"model": self.model, "prompt": text}
                    else:
                        payload = {"model": self.model, "input": text}

                    response = requests.post(
                        self._ollama_embedding_endpoint,
                        json=payload,
                        headers=headers,
                        timeout=60
                    )
                    response.raise_for_status()
                    result = response.json()
                    return result.get('embedding') or result.get('embeddings', [[]])[0]
                except Exception:
                    pass  # Fallback to discovery if cache fails

            # Discovery Phase
            endpoints = [
                f"{self.base_url}/api/embed",
                f"{self.base_url}/api/embeddings"
            ]

            for endpoint in endpoints:
                try:
                    time.sleep(0.01)
                    if "embeddings" in endpoint:
                        payload = {"model": self.model, "prompt": text}
                    else:
                        payload = {"model": self.model, "input": text}

                    response = requests.post(endpoint, json=payload, headers=headers, timeout=60)
                    if response.status_code in [404, 501]:
                        continue

                    response.raise_for_status()
                    result = response.json()

                    if 'embedding' in result or 'embeddings' in result:
                        self._ollama_embedding_endpoint = endpoint  # Cache it!
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

    # ========== MINIMAX CLOUD METHODS ==========

    def _minimax_cloud_generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Minimax Cloud API (Ollama.com cloud API)."""
        url = f"{self.base_url}/api/chat"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get('message', {}).get('content', '')
        except Exception as e:
            print(f"Error generating text with Minimax Cloud: {e}")
            return ""

    # ========== GEMINI METHODS ==========

    def _gemini_generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Gemini API."""
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.model}:generateContent?key={self.api_key}"
        )

        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": system_prompt}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {"contents": contents}

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            print(f"Error generating text: {e}")
            return ""

    def _gemini_parallel(self, texts: List[str]) -> List[List[float]]:
        """Gemini parallel embedding generation using threading."""
        def embed_single(text: str) -> List[float]:
            try:
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/"
                    f"models/{self.model}:embedContent?key={self.api_key}"
                )
                payload = {"content": {"parts": [{"text": text}]}}
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()
                return response.json()['embedding']['values']
            except Exception as e:
                print(f"Gemini error: {e}")
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

    # ========== MINIMAX CLOUD METHODS ==========

    def _minimax_cloud_generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Minimax Cloud API (Ollama.com cloud API)."""
        url = f"{self.base_url}/api/chat"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get('message', {}).get('content', '')
        except Exception as e:
            print(f"Error generating text with Minimax Cloud: {e}")
            return ""
