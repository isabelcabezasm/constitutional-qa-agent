from functools import cache, lru_cache

from azure.core.credentials import TokenCredential
from azure.identity import AzureCliCredential

from core.axiom_store import load_from_json
from core.azure_openai import azure_chat_openai_client
from core.paths import root
from core.qa_engine import QAEngine


@cache
def credential() -> TokenCredential:
    # Use WorkloadIdentityCredential if running in a Kubernetes environment
    return AzureCliCredential()


@cache
def azure_chat_openai():
    return azure_chat_openai_client(credential())


@cache
def axiom_store():
    return load_from_json((root() / "data/constitution.json").read_text())


@lru_cache(maxsize=32)
def qa_engine() -> QAEngine:
    return QAEngine(
        chat=azure_chat_openai()[0],
        deployment_name=azure_chat_openai()[1],
        axiom_store=axiom_store(),
    )
