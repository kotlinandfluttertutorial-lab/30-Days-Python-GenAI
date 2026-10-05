"""
Day 01 — LLM Client
===================
A provider-independent LLM client that works with:
- OpenAI (GPT-4o, GPT-4-mini)
- Anthropic (Claude)
- Groq (free, fast Llama/Mistral)
- Ollama (local, completely free)

This pattern is used throughout the program.

Run: python 02_llm_client.py
Requires: At least one API key in .env (or Ollama running)
"""

import os
import json
from typing import Optional, Any
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv

load_dotenv()


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    OLLAMA = "ollama"


@dataclass
class LLMMessage:
    """A single message in a conversation."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int

    @property
    def cost_estimate_usd(self) -> float:
        """Rough cost estimate based on provider and model."""
        costs = {
            "openai": {"input": 0.005, "output": 0.015},
            "anthropic": {"input": 0.003, "output": 0.015},
            "groq": {"input": 0.0001, "output": 0.0001},
            "ollama": {"input": 0.0, "output": 0.0},
        }
        provider_costs = costs.get(self.provider, {"input": 0.005, "output": 0.015})
        return (
            (self.input_tokens / 1000) * provider_costs["input"] +
            (self.output_tokens / 1000) * provider_costs["output"]
        )


class LLMClient:
    """
    Provider-independent LLM client.
    
    Abstracts over OpenAI, Anthropic, Groq, and Ollama.
    Use this pattern to avoid being locked into one provider.
    
    Usage:
        client = LLMClient(provider="groq")
        response = client.complete("What is RAG?")
        print(response.content)
    """

    def __init__(
        self,
        provider: str = "auto",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider, self.model = self._resolve_provider_and_model(provider, model)
        print(f"✓ LLM Client initialized: {self.provider} / {self.model}")

    def _resolve_provider_and_model(
        self, provider: str, model: Optional[str]
    ) -> tuple[str, str]:
        """Auto-detect available provider from environment."""
        if provider == "auto":
            # Try providers in order of preference
            if os.getenv("GROQ_API_KEY"):
                return "groq", model or "llama-3.1-8b-instant"
            elif os.getenv("OPENAI_API_KEY"):
                return "openai", model or "gpt-4o-mini"
            elif os.getenv("ANTHROPIC_API_KEY"):
                return "anthropic", model or "claude-3-haiku-20240307"
            else:
                # Fall back to Ollama (local, free)
                return "ollama", model or "llama3.2"
        
        default_models = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-haiku-20240307",
            "groq": "llama-3.1-8b-instant",
            "ollama": "llama3.2",
        }
        return provider, model or default_models.get(provider, "unknown")

    def complete(
        self,
        user_message: str,
        system_message: str = "You are a helpful AI assistant.",
    ) -> LLMResponse:
        """
        Send a single message and get a response.
        
        This is the most common pattern for AI applications.
        The system message sets the AI's behavior/role.
        """
        messages = [LLMMessage(role="user", content=user_message)]
        return self.chat(messages, system_message)

    def chat(
        self,
        messages: list[LLMMessage],
        system_message: str = "You are a helpful AI assistant.",
    ) -> LLMResponse:
        """
        Multi-turn conversation.
        
        This is the foundation for:
        - Chatbots (multiple turns)
        - Agents (tool calls in conversation history)
        - RAG (context added to first message)
        """
        if self.provider == "openai":
            return self._call_openai(messages, system_message)
        elif self.provider == "anthropic":
            return self._call_anthropic(messages, system_message)
        elif self.provider == "groq":
            return self._call_groq(messages, system_message)
        elif self.provider == "ollama":
            return self._call_ollama(messages, system_message)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_openai(
        self, messages: list[LLMMessage], system_message: str
    ) -> LLMResponse:
        """Call OpenAI API."""
        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        
        formatted_messages = [{"role": "system", "content": system_message}]
        formatted_messages += [
            {"role": m.role, "content": m.content} for m in messages
        ]

        response = client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,  # type: ignore
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=self.model,
            provider="openai",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

    def _call_anthropic(
        self, messages: list[LLMMessage], system_message: str
    ) -> LLMResponse:
        """Call Anthropic API."""
        try:
            import anthropic  # type: ignore
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        
        formatted_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]

        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_message,
            messages=formatted_messages,  # type: ignore
        )

        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            provider="anthropic",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

    def _call_groq(
        self, messages: list[LLMMessage], system_message: str
    ) -> LLMResponse:
        """
        Call Groq API (OpenAI-compatible).
        Groq offers FREE fast inference for Llama, Mistral, Gemma.
        Get your free API key at: https://console.groq.com
        """
        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )

        formatted_messages = [{"role": "system", "content": system_message}]
        formatted_messages += [
            {"role": m.role, "content": m.content} for m in messages
        ]

        response = client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,  # type: ignore
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=self.model,
            provider="groq",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

    def _call_ollama(
        self, messages: list[LLMMessage], system_message: str
    ) -> LLMResponse:
        """
        Call local Ollama server.
        Free, private, no API key required.
        Requires: ollama running with a model pulled.
        
        Install: https://ollama.com
        Run: ollama serve
        Pull model: ollama pull llama3.2
        """
        try:
            import requests  # type: ignore
        except ImportError:
            raise ImportError("Install requests: pip install requests")

        formatted_messages = [{"role": "system", "content": system_message}]
        formatted_messages += [
            {"role": m.role, "content": m.content} for m in messages
        ]

        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": self.model,
                    "messages": formatted_messages,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            
            content = data["message"]["content"]
            # Ollama doesn't always return token counts
            input_tokens = data.get("prompt_eval_count", 0)
            output_tokens = data.get("eval_count", 0)
            
            return LLMResponse(
                content=content,
                model=self.model,
                provider="ollama",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Cannot connect to Ollama. "
                "Is it running? Start with: ollama serve"
            )


def demo_llm_client() -> None:
    """Demonstrate the LLM client with AI engineering questions."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        DAY 01 — LLM CLIENT DEMO                          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    # Auto-detect available provider
    client = LLMClient(provider="auto", temperature=0.3)
    
    # Test questions
    questions = [
        "In one sentence, what is RAG in AI?",
        "In one sentence, what is the difference between an LLM and an AI Agent?",
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 50)
        
        try:
            response = client.complete(
                user_message=question,
                system_message=(
                    "You are an expert AI engineer. "
                    "Answer concisely and precisely. "
                    "Maximum 1-2 sentences."
                ),
            )
            print(f"Answer: {response.content}")
            print(f"Tokens: {response.input_tokens} in, {response.output_tokens} out")
            print(f"Cost: ${response.cost_estimate_usd:.6f}")
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure you have an API key in .env or Ollama running")


def demo_no_api_key() -> None:
    """Show LLM client structure without actually calling an API."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     LLM CLIENT STRUCTURE (No API Key Demo)               ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    print("\nThe LLMClient class provides:")
    print("  - LLMClient(provider='auto') — auto-detects available API")
    print("  - .complete(message) — single turn")
    print("  - .chat(messages) — multi-turn")
    print("  - LLMResponse.content — the text response")
    print("  - LLMResponse.input_tokens — tokens used for input")
    print("  - LLMResponse.output_tokens — tokens used for output")
    print("  - LLMResponse.cost_estimate_usd — estimated cost")
    
    print("\nSupported providers:")
    print("  - 'openai'    → Set OPENAI_API_KEY in .env")
    print("  - 'anthropic' → Set ANTHROPIC_API_KEY in .env")
    print("  - 'groq'      → Set GROQ_API_KEY in .env (FREE)")
    print("  - 'ollama'    → Run 'ollama serve' locally (FREE)")
    
    print("\nTo get a FREE API key:")
    print("  → Go to https://console.groq.com")
    print("  → Sign up, create API key")
    print("  → Add to .env: GROQ_API_KEY=gsk_...")
    
    print("\nTo use local Ollama (completely free):")
    print("  → Install from https://ollama.com")
    print("  → Run: ollama pull llama3.2")
    print("  → Run: ollama serve")
    print("  → Then: LLMClient(provider='ollama')")

    # Demonstrate message structure
    print("\nMessage structure for multi-turn chat:")
    messages = [
        LLMMessage(role="user", content="What is RAG?"),
        LLMMessage(role="assistant", content="RAG is Retrieval-Augmented Generation..."),
        LLMMessage(role="user", content="Can you give an example?"),
    ]
    print(json.dumps(
        [{"role": m.role, "content": m.content[:30] + "..."} for m in messages],
        indent=2
    ))


if __name__ == "__main__":
    # Check if any API key is available
    has_api_key = any([
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY"),
        os.getenv("GROQ_API_KEY"),
    ])
    
    if has_api_key:
        demo_llm_client()
    else:
        print("No API key found. Running structural demo instead.")
        print("To use a real LLM, add an API key to .env\n")
        demo_no_api_key()
