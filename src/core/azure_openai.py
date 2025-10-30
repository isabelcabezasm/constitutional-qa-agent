"""
Azure OpenAI dependency management module.

This module provides lazy-loaded dependencies for Azure OpenAI services

Prerequisites: - Azure CLI installed and configured (for CLI authentication) -
Valid Azure OpenAI deployment and configuration

"""

from typing import Final

from azure.core.credentials import TokenCredential
from azure.identity import get_bearer_token_provider
from openai import AzureOpenAI
from pydantic import BaseModel, HttpUrl

from core.env_values import env_values

COGN_SERVICES_SCOPE: Final = "https://cognitiveservices.azure.com/.default"


class AzureOpenAIConfig(BaseModel):
    """Configuration class for Azure OpenAI service.

    This class manages the configuration parameters required to connect to and
    interact with Azure OpenAI services, including deployment name, API
    version, and endpoint URL.

    Attributes:
        deployment_name (str): The name of the Azure OpenAI deployment to use.
        api_version (str): The API version for the Azure OpenAI service.
        endpoint (HttpUrl): The base URL endpoint for the Azure OpenAI service.

    Example:
        Creating a config from environment variables:

        >>> config = AzureOpenAIConfig.from_env()
        >>> print(config.deployment_name)
        'my-gpt-deployment'

        Creating a config manually:

        >>> config = AzureOpenAIConfig(
        ...     deployment_name="my-deployment",
        ...     api_version="2023-12-01-preview",
        ...     endpoint="https://my-resource.openai.azure.com/"
        ... )
    """

    deployment_name: str
    api_version: str
    endpoint: HttpUrl

    @classmethod
    def from_env(cls, /) -> "AzureOpenAIConfig":
        """
        Create an AzureOpenAIConfig instance from environment variables.

        This class method validates and extracts Azure OpenAI configuration from
        environment variables, ensuring all required fields are present and properly
        formatted.

        Returns:
            AzureOpenAIConfig: A new instance configured with values from environment
                variables including deployment name, API version, and endpoint URL.

        Raises:
            ValidationError: If any required environment variables are missing or
                have invalid values (e.g., malformed URL for endpoint).

        Environment Variables Required:
            AZURE_OPENAI_DEPLOYMENT_NAME: The name of the Azure OpenAI deployment
            OPENAI_API_VERSION: The API version to use for requests
            AZURE_OPENAI_ENDPOINT: The base URL endpoint for the Azure OpenAI service
        """

        class InternalConfig(BaseModel):
            AZURE_OPENAI_DEPLOYMENT_NAME: str
            OPENAI_API_VERSION: str
            AZURE_OPENAI_ENDPOINT: HttpUrl

        validated_env = InternalConfig.model_validate(env_values())

        return cls(
            deployment_name=validated_env.AZURE_OPENAI_DEPLOYMENT_NAME,
            api_version=validated_env.OPENAI_API_VERSION,
            endpoint=validated_env.AZURE_OPENAI_ENDPOINT,
        )


def azure_chat_openai_client(
    token_credential: TokenCredential,
    /,
    *,
    config: AzureOpenAIConfig | None = None,
) -> AzureOpenAI:
    """
    Create an Azure OpenAI client with token-based authentication.

    Args:
        token_credential: Azure token credential for authentication config:
        Optional Azure OpenAI configuration, defaults to env-based
                config

    Returns:
        Configured AzureOpenAI client instance
    """

    config = config or AzureOpenAIConfig.from_env()

    client = AzureOpenAI(
        api_version=config.api_version,
        azure_endpoint=str(config.endpoint),
        azure_ad_token_provider=get_bearer_token_provider(
            token_credential, COGN_SERVICES_SCOPE
        ),
    )
    return client


def create_qa_engine_with_config(
    token_credential: TokenCredential,
    /,
    *,
    config: AzureOpenAIConfig | None = None,
) -> tuple[AzureOpenAI, str]:
    """
    Create an Azure OpenAI client and return both client and deployment name.

    This is useful for creating a QA engine that needs both the client and
    the deployment name.

    Args:
        token_credential: Azure token credential for authentication
        config: Optional Azure OpenAI configuration, defaults to env-based config

    Returns:
        Tuple of (AzureOpenAI client, deployment_name)
    """
    config = config or AzureOpenAIConfig.from_env()
    client = azure_chat_openai_client(token_credential, config=config)
    return client, config.deployment_name
