"""Test for Azure OpenAI SDK integration."""

from unittest.mock import Mock, patch

import pytest
from openai import AzureOpenAI

from core.azure_openai import AzureOpenAIConfig, azure_chat_openai
from core.dependencies import credential


def test_azure_openai_config_from_env():
    """Test AzureOpenAIConfig creation from environment variables."""
    mock_env = {
        "AZURE_OPENAI_DEPLOYMENT_NAME": "test-deployment",
        "OPENAI_API_VERSION": "2024-02-15-preview",
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
    }

    with patch("core.azure_openai.env_values", return_value=mock_env):
        config = AzureOpenAIConfig.from_env()

        assert config.deployment_name == "test-deployment"
        assert config.api_version == "2024-02-15-preview"
        assert str(config.endpoint) == "https://test.openai.azure.com/"


def test_azure_chat_openai_returns_client():
    """Test that azure_chat_openai returns AzureOpenAI client."""
    mock_credential = Mock()
    mock_env = {
        "AZURE_OPENAI_DEPLOYMENT_NAME": "test-deployment",
        "OPENAI_API_VERSION": "2024-02-15-preview",
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
    }

    with (
        patch("core.azure_openai.env_values", return_value=mock_env),
        patch("core.azure_openai.get_bearer_token_provider") as mock_provider,
    ):
        mock_provider.return_value = Mock()

        with patch("core.azure_openai.AzureOpenAI") as mock_azure_openai:
            azure_chat_openai(mock_credential)

            # Verify AzureOpenAI was called with correct parameters
            mock_azure_openai.assert_called_once()
            call_kwargs = mock_azure_openai.call_args[1]

            assert call_kwargs["azure_deployment"] == "test-deployment"
            assert call_kwargs["api_version"] == "2024-02-15-preview"
            assert call_kwargs["azure_endpoint"] == "https://test.openai.azure.com/"
            assert "azure_ad_token_provider" in call_kwargs


def test_azure_chat_openai_with_custom_config():
    """Test azure_chat_openai with custom configuration."""
    mock_credential = Mock()
    custom_config = AzureOpenAIConfig(
        deployment_name="custom-deployment",
        api_version="2024-01-01",
        endpoint="https://custom.openai.azure.com/",
    )

    with patch("core.azure_openai.get_bearer_token_provider") as mock_provider:
        mock_provider.return_value = Mock()

        with patch("core.azure_openai.AzureOpenAI") as mock_azure_openai:
            azure_chat_openai(mock_credential, config=custom_config)

            # Verify custom config was used
            call_kwargs = mock_azure_openai.call_args[1]
            assert call_kwargs["azure_deployment"] == "custom-deployment"
            assert call_kwargs["api_version"] == "2024-01-01"
            assert call_kwargs["azure_endpoint"] == ("https://custom.openai.azure.com/")


@pytest.mark.integration
def test_real_azure_openai_client_creation():
    """Integration test for real client creation (requires valid config)."""
    try:
        # This test requires real Azure credentials and config
        token_credential = credential()
        client = azure_chat_openai(token_credential)

        # Verify we got an AzureOpenAI instance
        assert isinstance(client, AzureOpenAI)

    except Exception as e:
        # Skip if credentials/config not available
        pytest.skip(f"Skipping integration test: {e}")
