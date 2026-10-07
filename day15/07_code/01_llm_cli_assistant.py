"""
Day 15 — LLM CLI Assistant
============================
Production-quality LLM client with:
- Multiple provider support (auto-detect)
- Streaming responses
- Conversation history management
- Token counting and cost estimation
- Context window management

Run: python 01_llm_cli_assistant.py
Requires: At least one API key in .env, or Ollama running
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Generator, AsyncGenerator

from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────
# TOKEN UTILITIES
# ─────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """~4 chars per token (BPE heuristic)."""
    return max(1, len(text) // 4)


COST_PER_1K = {
    "gpt-4o":                  {"input": 0.005, "output": 0.015},
    "gpt-4o-mini":             {"input": 0.000150, "output": 0.000600},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "llama-3.1-70b-versatile": {"input": 0.00059, "output": 0.00079},
    "llama3.2":                {"input": 0.0, "output": 0.0},  # local
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    costs = COST_PER_1K.get(model, {"input": 0.005, "output": 0.015})
    return (input_tokens / 1000) * costs["input"] + (output_tokens / 1000) * costs["output"]


# ─────────────────────────────────────────────────────────
# CONVERSATION MANAGER
# ─────────────────────────────────────────────────────────

@dataclass
class ConversationManager:
    """Manages conversation history with context window limits."""
    system_prompt: str = "You are a helpful AI assistant specializing in AI engineering."
    max_context_tokens: int = 100_000
    messages: list[dict[str, str]] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self._trim_if_needed()

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def _trim_if_needed(self) -> None:
        """Remove oldest messages if context is too long."""
        while self._current_tokens() > self.max_context_tokens and len(self.messages) > 2:
            self.messages.pop(0)  # Remove oldest

    def _current_tokens(self) -> int:
        all_text = self.system_prompt + " ".join(m["content"] for m in self.messages)
        return estimate_tokens(all_text)

    def to_api_messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt}] + self.messages

    def clear(self) -> None:
        self.messages = []
        print("Conversation history cleared.")

    def stats(self) -> None:
        print(f"\n  Messages: {len(self.messages)}")
        print(f"  Context tokens: ~{self._current_tokens():,}")
        print(f"  Session input tokens: {self.total_input_tokens:,}")
        print(f"  Session output tokens: {self.total_output_tokens:,}")
        print(f"  Session cost: ${self.total_cost_usd:.6f}")


# ─────────────────────────────────────────────────────────
# LLM CLIENT
# ─────────────────────────────────────────────────────────

class LLMClient:
    """Provider-independent LLM client with streaming."""

    def __init__(self, temperature: float = 0.7, max_tokens: int = 1024) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider, self.model = self._detect_provider()
        print(f"✓ Provider: {self.provider} | Model: {self.model}")

    def _detect_provider(self) -> tuple[str, str]:
        if os.getenv("GROQ_API_KEY"):
            return "groq", "llama-3.1-70b-versatile"
        if os.getenv("OPENAI_API_KEY"):
            return "openai", "gpt-4o-mini"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic", "claude-3-5-sonnet-20241022"
        return "ollama", "llama3.2"

    def stream(self, messages: list[dict]) -> Generator[str, None, dict]:
        """Stream response tokens. Yields tokens, returns usage dict."""
        if self.provider in ("openai", "groq"):
            yield from self._stream_openai(messages)
        elif self.provider == "anthropic":
            yield from self._stream_anthropic(messages)
        elif self.provider == "ollama":
            yield from self._stream_ollama(messages)

    def _stream_openai(self, messages: list[dict]) -> Generator[str, None, None]:
        from openai import OpenAI

        if self.provider == "groq":
            client = OpenAI(
                api_key=os.environ["GROQ_API_KEY"],
                base_url="https://api.groq.com/openai/v1",
            )
        else:
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        stream = client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    def _stream_anthropic(self, messages: list[dict]) -> Generator[str, None, None]:
        import anthropic

        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        chat_messages = [m for m in messages if m["role"] != "system"]

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        with client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=chat_messages,  # type: ignore
        ) as stream:
            for text in stream.text_stream:
                yield text

    def _stream_ollama(self, messages: list[dict]) -> Generator[str, None, None]:
        import requests

        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "options": {"temperature": self.temperature},
                },
                stream=True,
                timeout=120,
            )
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if content := data.get("message", {}).get("content"):
                        yield content
                    if data.get("done"):
                        break
        except requests.exceptions.ConnectionError:
            yield "\n[Error: Cannot connect to Ollama. Run: ollama serve]"


# ─────────────────────────────────────────────────────────
# CLI INTERFACE
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AI engineering tutor helping a software engineer
transition into AI Engineering. You specialize in:
- LLMs, RAG, Agents, Embeddings, Vector Databases
- FastAPI, Docker, Python
- Production AI systems

Be concise and practical. Give code examples when helpful.
When asked about concepts, explain at a senior engineer level."""

COMMANDS = {
    "/clear":  "Clear conversation history",
    "/stats":  "Show token usage and cost",
    "/model":  "Show current model",
    "/help":   "Show this help",
    "/quit":   "Exit",
}


def print_banner() -> None:
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║        DAY 15 — LLM CLI ASSISTANT                        ║")
    print("║        Type /help for commands, /quit to exit            ║")
    print("╚══════════════════════════════════════════════════════════╝\n")


def handle_command(cmd: str, conv: ConversationManager, client: LLMClient) -> bool:
    """Handle CLI commands. Returns True if should continue."""
    cmd = cmd.strip().lower()
    if cmd == "/quit":
        print("\nGoodbye! Session stats:")
        conv.stats()
        return False
    elif cmd == "/clear":
        conv.clear()
    elif cmd == "/stats":
        conv.stats()
    elif cmd == "/model":
        print(f"  Provider: {client.provider} | Model: {client.model}")
    elif cmd == "/help":
        print("\n  Commands:")
        for cmd_name, desc in COMMANDS.items():
            print(f"    {cmd_name:<12} {desc}")
    else:
        print(f"  Unknown command. Type /help for available commands.")
    return True


def chat_loop() -> None:
    print_banner()

    try:
        client = LLMClient(temperature=0.7, max_tokens=1000)
    except Exception as e:
        print(f"Failed to initialize LLM client: {e}")
        print("Make sure you have an API key in .env or Ollama running")
        return

    conv = ConversationManager(system_prompt=SYSTEM_PROMPT)

    print("Ask me anything about AI engineering. Try:")
    print("  'What is RAG and how does it work?'")
    print("  'Explain the transformer attention mechanism'")
    print("  'How do I reduce LLM costs in production?'\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nInterrupted. Exiting.")
            conv.stats()
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            if not handle_command(user_input, conv, client):
                break
            continue

        # Add user message
        conv.add_user(user_input)

        # Stream response
        print("AI: ", end="", flush=True)
        start = time.perf_counter()
        full_response = ""

        try:
            for token in client.stream(conv.to_api_messages()):
                print(token, end="", flush=True)
                full_response += token
        except Exception as e:
            print(f"\n[Error: {e}]")
            conv.messages.pop()  # Remove failed user message
            continue

        elapsed = time.perf_counter() - start
        print()  # New line after response

        # Track stats
        in_tokens = estimate_tokens(user_input)
        out_tokens = estimate_tokens(full_response)
        cost = estimate_cost(client.model, in_tokens, out_tokens)

        conv.add_assistant(full_response)
        conv.total_input_tokens += in_tokens
        conv.total_output_tokens += out_tokens
        conv.total_cost_usd += cost

        print(f"  [~{out_tokens} tokens, {elapsed:.1f}s, ${cost:.5f}]\n")


if __name__ == "__main__":
    chat_loop()
