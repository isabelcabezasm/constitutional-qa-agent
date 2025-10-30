#!/bin/bash

# Development container setup script for eval-loop project
set -e

echo "🚀 Setting up eval-loop development environment..."

# Ensure uv is in PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Install project dependencies
echo "� Installing Python dependencies..."
cd /workspaces/eval-loop

# Check if uv is available, if not install it
if ! command -v uv &> /dev/null; then
    echo "🐍 Installing uv (Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Sync dependencies
uv sync

# Install pre-commit hooks (if .pre-commit-config.yaml exists)
if [ -f ".pre-commit-config.yaml" ]; then
    echo "🪝 Installing pre-commit hooks..."
    uv run pre-commit install
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p runs
mkdir -p logs
mkdir -p .vscode

# Set proper permissions
echo "🔒 Setting permissions..."
chmod +x /workspaces/eval-loop/.devcontainer/post_create.sh

# Create a sample .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "🔐 Creating sample .env file..."
    cat > .env << EOF
# Azure Configuration
AZURE_CLIENT_ID=your_client_id_here
AZURE_CLIENT_SECRET=your_client_secret_here
AZURE_TENANT_ID=your_tenant_id_here

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Application Settings
DEBUG=true
LOG_LEVEL=INFO
EOF
fi

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Configure your .env file with actual credentials"
echo "2. Run 'uv run python -m eval.main --help' to see available commands"
echo "3. Run tests with 'uv run pytest'"
echo "4. Start coding! 🚀"
echo ""
echo "📝 Available commands:"
echo "  - uv sync                    # Install/update dependencies"
echo "  - uv run pytest            # Run tests"
echo "  - uv run ruff check        # Lint code"
echo "  - uv run ruff format       # Format code"
echo "  - uv run pyright           # Type checking"
echo "  - uv run python -m eval.main  # Run evaluation"