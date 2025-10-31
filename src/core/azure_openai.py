"""
Azure OpenAI dependency management module.

This module provides lazy-loaded dependencies for Azure OpenAI services
using the Microsoft Agent Framework.

Prerequisites:
- Azure CLI installed and configured (for CLI authentication)
- Valid Azure OpenAI deployment and configuration via environment variables

Environment Variables:
The AzureOpenAIChatClient automatically reads from these environment variables:
- AZURE_OPENAI_CHAT_DEPLOYMENT_NAME: The name of the Azure OpenAI deployment
  (required)
- AZURE_OPENAI_ENDPOINT: The base URL endpoint for the Azure OpenAI service
  (required)
- AZURE_OPENAI_API_VERSION: The API version to use for requests
  (optional, uses default if not set)

For more information, see:
https://github.com/microsoft/agent-framework/blob/main/python/samples/getting_started/chat_client/azure_chat_client.py
"""

from agent_framework.azure import AzureOpenAIChatClient
from azure.core.credentials import TokenCredential


def azure_chat_openai_client(
    token_credential: TokenCredential,
    /,
) -> AzureOpenAIChatClient:
    """
    Create an Azure OpenAI chat client using the Microsoft Agent Framework.

    The client automatically reads configuration from environment variables:
    - AZURE_OPENAI_CHAT_DEPLOYMENT_NAME (required)
    - AZURE_OPENAI_ENDPOINT (required)
    - AZURE_OPENAI_API_VERSION (optional)

    Args:
        token_credential: Azure token credential for authentication

    Returns:
        Configured AzureOpenAIChatClient instance
    """
    return AzureOpenAIChatClient(credential=token_credential)
