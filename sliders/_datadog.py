"""Datadog Agent (LLM) Observability bootstrap.

Importing this module loads local environment variables (so ``DD_*`` settings
from a ``.env`` file are picked up) and then initializes ``ddtrace`` auto
instrumentation. It must be imported before any LLM SDK (e.g. ``langchain`` or
``openai``) so those libraries get patched for Agent Observability.
"""

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

import ddtrace.auto  # noqa: E402,F401
