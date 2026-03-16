#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Interactive terminal chat client for a vLLM server.

Connects to a running vLLM instance (local or multi-node cluster) via its
OpenAI-compatible API and lets you have a streaming conversation in the
terminal.

Usage
-----
Start your vLLM server first (single node or multi-node cluster), e.g.:

    vllm serve Qwen/Qwen3-8B --tensor-parallel-size 4

Then run this script:

    python chat.py                          # defaults: localhost:8000
    python chat.py --host 192.168.1.10      # remote head node
    python chat.py --host 192.168.1.10 --port 8000 --model Qwen/Qwen3-8B
    python chat.py --system "You are a pirate. Respond only in pirate speak."

Commands available during the chat session:
    /exit  or  /quit  — end the session
    /clear            — clear conversation history (start fresh)
    /history          — print the conversation so far
    /model            — show which model is being used
    /no_think         — disable extended thinking (Qwen3 models, thinking on by default)
    /think            — re-enable extended thinking
    /help             — show this list
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package is required. Install with: pip install openai")
    sys.exit(1)


# ---------------------------------------------------------------------------
# GFLOP estimation helpers
# ---------------------------------------------------------------------------

class ModelFlopsEstimator:
    """Estimates compute work (GFLOPs) for transformer forward passes.

    Architecture parameters are fetched once from the Hugging Face Hub
    config.json for the served model.  If the fetch fails we fall back to
    a parameter-count-based approximation using the rule of thumb:

        FLOPs ≈ 6 × N_params × N_tokens

    For a detailed derivation of the per-layer formula see:
      https://arxiv.org/abs/2001.08361  (Kaplan et al., §2)

    The full formula used when config data is available:
      FLOPs per token ≈ 2 × L × (12 × H² + 4 × H × FFN_dim)
    where L = num_hidden_layers, H = hidden_size, FFN_dim = intermediate_size.

    In a pipeline-parallel cluster each node processes (L / PP) layers, so
    per-node GFLOPs = total_GFLOPs / num_nodes.
    """

    def __init__(self, model_id: str, base_url: str) -> None:
        self.model_id = model_id
        self.base_url = base_url  # e.g. "http://host:port/v1"
        self._server_root = base_url.rstrip("/").removesuffix("/v1")

        # Architecture params (filled by _fetch_model_config)
        self._num_layers: int | None = None
        self._hidden_size: int | None = None
        self._intermediate_size: int | None = None
        self._num_params: int | None = None  # fallback

        # Parallelism info (filled by _fetch_parallelism)
        self._num_nodes: int = 1

        # Cumulative totals across the whole session
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0

        self._init()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init(self) -> None:
        self._fetch_model_config()
        # Parallelism info is fetched lazily on first record_and_print() call,
        # so startup doesn't add an extra HTTP round-trip that could compete
        # with in-flight inference requests on the server's event loop.

    def _fetch_model_config(self) -> None:
        """Try to load hidden_size / num_layers / intermediate_size from HF."""
        # Normalise: strip leading "./" or absolute paths so we only pass
        # the HF repo id (e.g. "Qwen/Qwen3-8B").
        repo_id = self.model_id.lstrip("./")
        url = f"https://huggingface.co/{repo_id}/resolve/main/config.json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "chat.py/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                cfg = json.loads(resp.read())
            self._hidden_size = cfg.get("hidden_size")
            self._intermediate_size = cfg.get("intermediate_size")
            self._num_layers = (
                cfg.get("num_hidden_layers")
                or cfg.get("n_layer")
                or cfg.get("num_layers")
            )
        except Exception:
            # HF unreachable or local path – fall back to param-count heuristic
            pass

    def _fetch_parallelism(self) -> None:
        """Scrape /metrics to detect pipeline_parallel_size → num_nodes.

        vLLM's Prometheus metrics include a gauge labelled with
        ``pipeline_parallel_size`` and ``tensor_parallel_size``.  We look
        for the line:

            vllm:num_requests_running{...,pipeline_parallel_size="N",...} ...

        and use pipeline_parallel_size as a proxy for number of nodes.
        If the metric is absent we default to 1.
        """
        try:
            url = f"{self._server_root}/metrics"
            req = urllib.request.Request(url, headers={"User-Agent": "chat.py/1.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                text = resp.read().decode()
            for line in text.splitlines():
                if "pipeline_parallel_size" in line and not line.startswith("#"):
                    # Extract pipeline_parallel_size="N"
                    m = re.search(r'pipeline_parallel_size="(\d+)"', line)
                    if m:
                        self._num_nodes = max(1, int(m.group(1)))
                        break
        except Exception:
            pass  # keep default of 1

    # ------------------------------------------------------------------
    # FLOP estimation
    # ------------------------------------------------------------------

    def _flops_per_token(self) -> float | None:
        """Return estimated FLOPs for one token through the full model."""
        if (self._num_layers is not None
                and self._hidden_size is not None
                and self._intermediate_size is not None):
            H = self._hidden_size
            FFN = self._intermediate_size
            L = self._num_layers
            # Attention: 4 × matmuls of size H×H (Q,K,V proj + out proj)
            # → 2 × 4 × H² per layer
            # FFN: 2 matmuls of H×FFN → 2 × 2 × H × FFN per layer
            # Factor-of-2 for multiply-add
            return 2.0 * L * (8.0 * H * H + 4.0 * H * FFN)
        return None

    def record_and_print(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Update running totals and print per-node GFLOP debug line."""
        # Lazy-fetch parallelism info once, after the first inference completes,
        # so we don't compete with in-flight requests at startup.
        if self._num_nodes == 1:
            self._fetch_parallelism()

        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens

        total_tokens = self._total_prompt_tokens + self._total_completion_tokens
        this_tokens = prompt_tokens + completion_tokens
        num_nodes = self._num_nodes

        fpt = self._flops_per_token()
        if fpt is not None:
            total_gflops = fpt * total_tokens / 1e9
            this_gflops = fpt * this_tokens / 1e9
        else:
            # Rough fallback: can't compute without architecture info
            print(
                f"  [debug] Could not fetch model config – "
                f"GFLOP estimate unavailable "
                f"(tokens this turn: {this_tokens:,}, total: {total_tokens:,})"
            )
            return

        per_node_total = total_gflops / num_nodes
        per_node_turn  = this_gflops  / num_nodes

        node_str = (
            f"{num_nodes} node{'s' if num_nodes != 1 else ''}"
        )
        print(
            f"  [debug] GFLOPs — this turn: {this_gflops:,.1f} total "
            f"({per_node_turn:,.1f}/node) | "
            f"session total: {total_gflops:,.1f} "
            f"({per_node_total:,.1f}/node) "
            f"[{node_str}, tokens this turn: {prompt_tokens}+{completion_tokens}]"
        )


COMMANDS = {
    "/exit":    "Exit the chat session",
    "/think":   "Enable extended thinking (Qwen3/QwQ models)",
    "/no_think": "Disable extended thinking for immediate streaming",
    "/quit":    "Exit the chat session",
    "/clear":   "Clear conversation history and start fresh",
    "/history": "Print the full conversation history",
    "/model":   "Show the model currently being used",
    "/help":    "Show this help message",
}


def print_help() -> None:
    print("\nAvailable commands:")
    for cmd, desc in COMMANDS.items():
        print(f"  {cmd:<12} {desc}")
    print()


def print_history(messages: list[dict]) -> None:
    print("\n── Conversation History ──────────────────────────────────────")
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        print(f"\n[{role}]\n{content}")
    print("\n──────────────────────────────────────────────────────────────\n")


def stream_response(client: OpenAI, model: str,
                    messages: list[dict],
                    temperature: float,
                    max_tokens: int | None,
                    enable_thinking: bool = True) -> tuple[str, int, int]:
    """Send messages to the vLLM server and stream the response token by token.

    Returns a tuple of (full_response_text, prompt_tokens, completion_tokens).

    stream_options={"include_usage": True} is sent so vLLM appends a final
    usage-only chunk (choices=[]) before [DONE].  The OpenAI client handles
    this transparently; we capture it to get exact token counts for GFLOP
    estimation without any separate API call.
    """
    kwargs: dict = dict(
        model=model,
        messages=messages,
        stream=True,
        temperature=temperature,
        # include_usage gives us exact token counts for GFLOP estimation.
        # The usage chunk arrives before [DONE] so it does not cause hangs.
        stream_options={"include_usage": True},
        # Pass enable_thinking via chat_template_kwargs (the correct vLLM path).
        # Top-level extra_body fields are silently ignored for this setting.
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    print("\nAssistant: ", end="", flush=True)
    full_response = []
    prompt_tokens = 0
    completion_tokens = 0

    try:
        stream = client.chat.completions.create(**kwargs)
        for chunk in stream:
            # vLLM sets usage on the final chunk (finish_reason != None)
            # even without stream_options, so we capture it here whenever
            # it appears rather than relying on a separate trailing chunk.
            if chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0
            if chunk.choices:
                delta = chunk.choices[0].delta
                token = getattr(delta, "content", None)
                if token:
                    print(token, end="", flush=True)
                    full_response.append(token)
    except KeyboardInterrupt:
        # Allow Ctrl-C mid-stream to stop generation gracefully
        print("\n[Generation interrupted]", flush=True)

    print()  # newline after streamed response
    return "".join(full_response), prompt_tokens, completion_tokens


def resolve_model(client: OpenAI, requested_model: str | None) -> str:
    """Return the model to use. If none requested, pick the first available."""
    models = client.models.list()
    available = [m.id for m in models.data]

    if not available:
        print("Error: No models are available on the server.")
        sys.exit(1)

    if requested_model is None:
        return available[0]

    if requested_model not in available:
        print(f"Warning: Model '{requested_model}' not found on server.")
        print(f"Available models: {', '.join(available)}")
        print(f"Using '{available[0]}' instead.\n")
        return available[0]

    return requested_model


def run_chat(args: argparse.Namespace) -> None:
    base_url = f"http://{args.host}:{args.port}/v1"
    client = OpenAI(api_key=args.api_key, base_url=base_url)

    # Verify connection and resolve model
    try:
        model = resolve_model(client, args.model)
    except Exception as e:
        print(f"Error: Could not connect to vLLM server at {base_url}")
        print(f"  {e}")
        print("\nMake sure your vLLM server is running, e.g.:")
        print("  vllm serve <model> --host 0.0.0.0 --port 8000")
        sys.exit(1)

    # Set up GFLOP estimator (fetches model config from HF + parallelism from /metrics)
    flops_estimator = ModelFlopsEstimator(model_id=model, base_url=base_url)

    # Thinking mode: ON by default, disabled with --no-think flag.
    # Qwen3 and similar models generate a <think>...</think> chain-of-thought
    # before each visible response.  Use --no-think or /no_think to skip it.
    enable_thinking: bool = not args.no_think
    if not enable_thinking:
        if "qwen3" in model.lower() or "qwq" in model.lower():
            print(f"  [info] Thinking mode OFF for {model}. "
                  f"Use /think to re-enable.")

    # Build initial message history
    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    # ── Welcome banner ────────────────────────────────────────────────────────
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║            vLLM Interactive Chat Client                     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Server : {base_url}")
    print(f"  Model  : {model}")
    if args.system:
        print(f"  System : {args.system[:60]}{'...' if len(args.system) > 60 else ''}")
    print("\nType /help for commands, or /exit to quit.\n")

    # ── Main chat loop ────────────────────────────────────────────────────────
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # ── Handle commands ───────────────────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]

            if cmd in ("/exit", "/quit"):
                print("Goodbye!")
                break

            elif cmd == "/clear":
                messages = []
                if args.system:
                    messages.append({"role": "system", "content": args.system})
                print("Conversation history cleared.\n")
                continue

            elif cmd == "/history":
                if len(messages) == 0 or (
                        len(messages) == 1
                        and messages[0]["role"] == "system"):
                    print("No conversation history yet.\n")
                else:
                    print_history(messages)
                continue

            elif cmd == "/model":
                print(f"Current model: {model}\n")
                continue

            elif cmd == "/think":
                enable_thinking = True
                print("  Thinking mode enabled. The model will reason before responding.\n")
                continue

            elif cmd == "/no_think":
                enable_thinking = False
                print("  Thinking mode disabled. Responses will stream immediately.\n")
                continue

            elif cmd == "/help":
                print_help()
                continue

            else:
                print(f"Unknown command '{cmd}'. Type /help for available commands.\n")
                continue

        # ── Send message to the model ─────────────────────────────────────────
        messages.append({"role": "user", "content": user_input})

        try:
            response_text, prompt_tokens, completion_tokens = stream_response(
                client=client,
                model=model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                enable_thinking=enable_thinking,
            )
        except Exception as e:
            print(f"\nError communicating with server: {e}\n")
            # Remove the user message we just added since it wasn't answered
            messages.pop()
            continue

        if response_text:
            messages.append({"role": "assistant", "content": response_text})

        # Print per-node GFLOP debug line after every inference turn
        flops_estimator.record_and_print(prompt_tokens, completion_tokens)

        print()  # blank line between turns


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive terminal chat client for a vLLM server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Hostname or IP of the vLLM server (head node in a cluster).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port the vLLM server is listening on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use. Defaults to the first model available on the server.",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="System prompt to set the assistant's behaviour.",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        default=False,
        help=(
            "Disable extended thinking for Qwen3 and similar thinking models. "
            "Thinking is on by default. Use this to skip the chain-of-thought "
            "and get immediate token streaming. Can also be toggled mid-session "
            "with /think and /no_think."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0 = deterministic, higher = more creative).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate per response. "
             "Defaults to the server's limit.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="EMPTY",
        help="API key for the vLLM server (use 'EMPTY' if none is set).",
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run_chat(args)
