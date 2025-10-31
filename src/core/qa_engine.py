"""
QA Engine for constitutional AI assistant.

This module provides the QAEngine class that handles question-answering
using Azure OpenAI with constitution-based prompting via the Microsoft Agent Framework.
"""

import json

from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient

from core.axiom_store import Axiom, AxiomStore
from core.paths import root


class QAEngine:
    """
    Question-Answering engine for constitutional queries.

    This class handles the orchestration of prompts, constitution loading,
    and interaction with Azure OpenAI via the Microsoft Agent Framework to provide
    contextualized responses based on predefined axioms.

    Attributes:
        agent (ChatAgent): The chat agent for model inference.
        axiom_store (AxiomStore | None): Storage for axioms/constitution data.
    """

    def __init__(
        self,
        chat: AzureOpenAIChatClient,
        axiom_store: AxiomStore | None = None,
    ):
        """
        Initialize the QA Engine.

        Args:
            chat: Azure OpenAI chat client instance for model inference.
            axiom_store: Optional storage for axioms (defaults to loading from file).
        """
        self.chat = chat
        self.axiom_store = axiom_store
        self.agent: ChatAgent | None = None

    def _load_constitution_data(self) -> list[Axiom]:
        """
        Load constitution data from JSON file.

        Returns:
            List of axioms from the constitution JSON file.
        """
        constitution_file = root() / "data/constitution.json"
        with open(constitution_file, encoding="utf-8") as f:
            return [Axiom(**item) for item in json.load(f)]

    def _load_and_format_constitution(self) -> str:
        """
        Load the constitution template and format it with axiom data.

        Returns:
            Formatted constitution text with axioms.
        """
        # Load constitution template
        constitution_template_file = root() / "src/core/prompts/constitution.md"
        with open(constitution_template_file, encoding="utf-8") as f:
            template_content = f.read()

        # Load axiom data
        axioms = self.axiom_store or self._load_constitution_data()
        axiom_list = axioms.list() if isinstance(axioms, AxiomStore) else axioms

        # Format constitution by replacing template variables for each axiom
        formatted_constitution = ""
        for axiom in axiom_list:
            # Create a copy of the template for each axiom
            axiom_section = template_content

            # Replace template variables with axiom data
            # Map axiom fields to template variables
            replacements = {
                "{{ id }}": axiom.id,
                "{{ subject }}": axiom.subject,
                "{{ object }}": axiom.entity,  # Map entity to object
                "{{ link }}": axiom.trigger,  # Map trigger to link
                "{{ conditions }}": axiom.conditions,
                "{{ description }}": axiom.description,
                # Map category to amendments
                "{{ amendments }}": f"Category: {axiom.category}",
            }

            # Apply all replacements
            for placeholder, value in replacements.items():
                axiom_section = axiom_section.replace(placeholder, value)

            formatted_constitution += axiom_section + "\n"

        return formatted_constitution

    def _load_system_prompt(self) -> str:
        """
        Load the system prompt.

        Returns:
            System prompt text.
        """
        system_prompt_file = root() / "src/core/prompts/system_prompt.md"
        with open(system_prompt_file, encoding="utf-8") as f:
            return f.read()

    def _load_and_format_user_prompt(self, question: str) -> str:
        """
        Load and format the user prompt with constitution and question.

        Args:
            question: The user's question to be answered.

        Returns:
            Formatted user prompt with constitution and question.
        """
        # Load user prompt template
        user_prompt_file = root() / "src/core/prompts/user_prompt.md"
        with open(user_prompt_file, encoding="utf-8") as f:
            user_prompt_template = f.read()

        # Get formatted constitution
        constitution = self._load_and_format_constitution()

        # Format user prompt with constitution and question using string replacement
        formatted_prompt = user_prompt_template.replace(
            "{{ constitution }}", constitution
        )
        formatted_prompt = formatted_prompt.replace("{{ question }}", question)

        return formatted_prompt

    async def invoke(self, question: str) -> str:
        """
        Process a user question and generate a response using Azure OpenAI.

        This method:
        1. Loads and formats the constitution with axiom data
        2. Prepares the system prompt (used as agent instructions)
        3. Formats the user prompt with the question and constitution
        4. Creates a ChatAgent with system instructions
        5. Queries the Azure OpenAI model via the Agent Framework
        6. Returns the generated response

        Args:
            question: The user's question.

        Returns:
            The AI-generated response based on the constitution and prompts.
        """

        # Load system prompt to use as agent instructions
        system_prompt = self._load_system_prompt()

        # Create agent with system instructions if not already created
        # or if we need to update instructions
        if self.agent is None:
            self.agent = self.chat.create_agent(instructions=system_prompt)

        # Load and format user prompt with constitution and question
        user_prompt = self._load_and_format_user_prompt(question)

        # Use asyncio to run the async agent
        response = await self.agent.run(user_prompt)

        return str(response.text)
