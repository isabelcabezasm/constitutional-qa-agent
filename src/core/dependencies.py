from functools import cache, lru_cache

from agent_framework import ChatAgent
from azure.core.credentials import TokenCredential
from azure.identity import AzureCliCredential

from core.axiom_store import load_from_json
from core.azure_openai import azure_chat_openai_client
from core.paths import root
from core.qa_engine import QAEngine


@cache
def credential() -> TokenCredential:
    """Get Azure token credential for authentication."""
    # Use WorkloadIdentityCredential if running in a Kubernetes environment
    return AzureCliCredential()


@cache
def azure_chat_openai():
    """Create and cache the Azure OpenAI chat client."""
    return azure_chat_openai_client(credential())


@cache
def chat_agent() -> ChatAgent:
    """Create and cache the ChatAgent with system prompt."""
    system_prompt = (root() / "src/core/prompts/system_prompt.md").read_text()
    return azure_chat_openai().create_agent(instructions=system_prompt)


@cache
def axiom_store():
    """Load and cache the constitutional axioms from JSON data file."""
    return load_from_json((root() / "data/constitution.json").read_text())


@lru_cache(maxsize=32)
def qa_engine() -> QAEngine:
    """Create and cache the QA Engine with Agent Framework client."""
    return QAEngine(
        agent=chat_agent(),
        axiom_store=axiom_store(),
    )
