"""Test for Azure OpenAI SDK integration."""

from unittest.mock import Mock, patch

import pytest
from openai import AzureOpenAI
from pydantic import HttpUrl, ValidationError

from core.azure_openai import (
    COGN_SERVICES_SCOPE,
    AzureOpenAIConfig,
    azure_chat_openai_client,
)
from core.dependencies import credential

# test AzureOpenAIConfig


def test_config_creation_with_valid_data():
    """Test creating AzureOpenAIConfig with valid data."""
    config = AzureOpenAIConfig(
        deployment_name="test-deployment",
        api_version="2024-02-15-preview",
        endpoint=HttpUrl("https://test.openai.azure.com/"),
    )

    assert config.deployment_name == "test-deployment"
    assert config.api_version == "2024-02-15-preview"
    assert str(config.endpoint) == "https://test.openai.azure.com/"


def test_config_creation_with_invalid_endpoint():
    """Test creating AzureOpenAIConfig with invalid endpoint URL."""
    with pytest.raises(ValidationError):
        AzureOpenAIConfig(
            deployment_name="test-deployment",
            api_version="2024-02-15-preview",
            endpoint="invalid-url",  # type: ignore
        )


# test creating AzureOpenAI client from env


def test_config_from_env_success():
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


@pytest.mark.parametrize(
    "mock_env,missing_field",
    [
        (
            {
                "OPENAI_API_VERSION": "2024-02-15-preview",
                "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
                # Missing AZURE_OPENAI_DEPLOYMENT_NAME
            },
            "AZURE_OPENAI_DEPLOYMENT_NAME",
        ),
        (
            {
                "AZURE_OPENAI_DEPLOYMENT_NAME": "test-deployment",
                "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
                # Missing OPENAI_API_VERSION
            },
            "OPENAI_API_VERSION",
        ),
        (
            {
                "AZURE_OPENAI_DEPLOYMENT_NAME": "test-deployment",
                "OPENAI_API_VERSION": "2024-02-15-preview",
                # Missing AZURE_OPENAI_ENDPOINT
            },
            "AZURE_OPENAI_ENDPOINT",
        ),
    ],
)
def test_config_from_env_missing_required_fields(mock_env, missing_field):
    """Test AzureOpenAIConfig creation fails with missing required environment
    variables."""
    with patch("core.azure_openai.env_values", return_value=mock_env):
        with pytest.raises(ValidationError) as exc_info:
            AzureOpenAIConfig.from_env()

    assert missing_field in str(exc_info.value)


def test_config_from_env_invalid_endpoint():
    """Test AzureOpenAIConfig creation fails with invalid endpoint URL."""
    mock_env = {
        "AZURE_OPENAI_DEPLOYMENT_NAME": "test-deployment",
        "OPENAI_API_VERSION": "2024-02-15-preview",
        "AZURE_OPENAI_ENDPOINT": "invalid-url",
    }

    with patch("core.azure_openai.env_values", return_value=mock_env):
        with pytest.raises(ValidationError) as exc_info:
            AzureOpenAIConfig.from_env()

        assert "AZURE_OPENAI_ENDPOINT" in str(exc_info.value)


def test_azure_chat_openai_without_config():
    """Test azure_chat_openai_client without "AzureOpenAIConfig"."""
    mock_credential = Mock()

    # Create the mock config that should be returned by from_env
    mock_config = AzureOpenAIConfig(
        deployment_name="test-deployment",
        api_version="2023-01-01-preview",
        endpoint=HttpUrl("https://project.cognitiveservices.azure.com/"),
    )

    with patch("core.azure_openai.get_bearer_token_provider") as mock_provider:
        mock_token_provider = Mock()
        mock_provider.return_value = mock_token_provider

        with patch("core.azure_openai.AzureOpenAI") as mock_azure_openai:
            mock_client = Mock(spec=AzureOpenAI)
            mock_azure_openai.return_value = mock_client

            # Mock the from_env method specifically
            with patch(
                "core.azure_openai.AzureOpenAIConfig"
            ) as mock_azure_openai_config:
                mock_azure_openai_config.from_env.return_value = mock_config

                result, deployment_name = azure_chat_openai_client(mock_credential)

                # Verify from_env was called
                mock_azure_openai_config.from_env.assert_called_once()

                # Verify default config was used
                call_kwargs = mock_azure_openai.call_args[1]
                assert call_kwargs["api_version"] == "2023-01-01-preview"
                assert (
                    call_kwargs["azure_endpoint"]
                    == "https://project.cognitiveservices.azure.com/"
                )
                assert call_kwargs["azure_ad_token_provider"] == mock_token_provider

                # Verify get_bearer_token_provider was called correctly
                mock_provider.assert_called_once_with(
                    mock_credential, COGN_SERVICES_SCOPE
                )

                # Verify we got the mocked client back
                assert result == mock_client
                assert deployment_name == "test-deployment"


def test_azure_chat_openai_with_config():
    """Test azure_chat_openai_client with "AzureOpenAIConfig"."""
    mock_credential = Mock()
    custom_config = AzureOpenAIConfig(
        deployment_name="custom-deployment",
        api_version="2024-01-01",
        endpoint=HttpUrl("https://custom.openai.azure.com/"),
    )

    with patch("core.azure_openai.get_bearer_token_provider") as mock_provider:
        mock_token_provider = Mock()
        mock_provider.return_value = mock_token_provider

        with patch("core.azure_openai.AzureOpenAI") as mock_azure_openai:
            mock_client = Mock(spec=AzureOpenAI)
            mock_azure_openai.return_value = mock_client

            result, deployment_name = azure_chat_openai_client(
                mock_credential, config=custom_config
            )

            # Verify custom config was used
            call_kwargs = mock_azure_openai.call_args[1]
            assert call_kwargs["api_version"] == "2024-01-01"
            assert call_kwargs["azure_endpoint"] == "https://custom.openai.azure.com/"
            assert call_kwargs["azure_ad_token_provider"] == mock_token_provider

            # Verify get_bearer_token_provider was called correctly
            mock_provider.assert_called_once_with(mock_credential, COGN_SERVICES_SCOPE)

            # Verify we got the mocked client back
            assert result == mock_client
            assert deployment_name == "custom-deployment"


def test_azure_chat_openai_returns_azure_openai_instance():
    """Test that azure_chat_openai_client returns AzureOpenAI instance."""
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

        # Simply mock the AzureOpenAI constructor to return a mock instance
        with patch("core.azure_openai.AzureOpenAI") as mock_azure_openai:
            mock_instance = Mock(spec=AzureOpenAI)
            mock_azure_openai.return_value = mock_instance

            result, deployment_name = azure_chat_openai_client(mock_credential)

            # Verify we got back the mocked AzureOpenAI instance
            assert result == mock_instance
            assert hasattr(result, "chat")  # AzureOpenAI should have chat attribute
            assert deployment_name == "test-deployment"


def test_azure_chat_openai():
    """Test azure_chat_openai_client with default (env-based) config."""
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
        mock_token_provider = Mock()
        mock_provider.return_value = mock_token_provider

        with patch("core.azure_openai.AzureOpenAI") as mock_azure_openai:
            mock_client = Mock(spec=AzureOpenAI)
            mock_azure_openai.return_value = mock_client

            result, deployment_name = azure_chat_openai_client(mock_credential)

            # Verify AzureOpenAI was called with correct parameters
            mock_azure_openai.assert_called_once()
            call_kwargs = mock_azure_openai.call_args[1]

            assert call_kwargs["api_version"] == "2024-02-15-preview"
            assert call_kwargs["azure_endpoint"] == "https://test.openai.azure.com/"
            assert call_kwargs["azure_ad_token_provider"] == mock_token_provider

            # Verify get_bearer_token_provider was called correctly
            mock_provider.assert_called_once_with(mock_credential, COGN_SERVICES_SCOPE)

            # Verify we got the mocked client back
            assert result == mock_client
            assert deployment_name == "test-deployment"


def test_create_qa_engine_config_fallback():
    """Test that create_qa_engine_with_config creates config when None provided."""
    mock_credential = Mock()
    mock_env = {
        "AZURE_OPENAI_DEPLOYMENT_NAME": "fallback-deployment",
        "OPENAI_API_VERSION": "2024-02-15-preview",
        "AZURE_OPENAI_ENDPOINT": "https://fallback.openai.azure.com/",
    }

    with (
        patch("core.azure_openai.env_values", return_value=mock_env),
        patch("core.azure_openai.azure_chat_openai_client") as mock_client_func,
    ):
        mock_client = Mock(spec=AzureOpenAI)
        mock_client_func.return_value = mock_client

        # Call with config=None (explicit)
        _, deployment_name = azure_chat_openai_client(mock_credential, config=None)

        # Verify the function falls back to creating config from env
        assert deployment_name == "fallback-deployment"


@pytest.mark.integration
def test_real_azure_openai_client_creation():
    """Integration test for real client creation (requires valid config)."""
    try:
        # This test requires real Azure credentials and config
        token_credential = credential()
        client, deployment_name = azure_chat_openai_client(token_credential)

        # Verify we got an AzureOpenAI instance
        assert isinstance(client, AzureOpenAI)
        assert isinstance(deployment_name, str)
        assert len(deployment_name) > 0

    except (ValueError, ImportError, AttributeError) as e:
        # Skip if credentials/config not available
        pytest.skip(f"Skipping integration test: {e}")
