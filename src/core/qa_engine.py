"""
QA Engine for health insurance AI assistant.

This module provides the QAEngine class that handles question-answering
using Azure OpenAI with constitution-based prompting.
"""

import json

from openai import AzureOpenAI

from core.axiom_store import Axiom, AxiomStore
from core.paths import root


class QAEngine:
    """
    Question-Answering engine for health insurance queries.

    This class handles the orchestration of prompts, constitution loading,
    and interaction with Azure OpenAI to provide contextualized responses
    based on predefined axioms.

    Attributes:
        chat (AzureOpenAI): The Azure OpenAI client for model inference.
        deployment_name (str): The Azure OpenAI deployment name.
        axiom_store (AxiomStore | None): Storage for axioms/constitution data.
    """

    def __init__(
        self,
        chat: AzureOpenAI,
        deployment_name: str,
        axiom_store: AxiomStore | None = None,
    ):
        """
        Initialize the QA Engine.

        Args:
            chat: Azure OpenAI client instance for model inference.
            deployment_name: The Azure OpenAI deployment name to use for inference.
            axiom_store: Optional storage for axioms (defaults to loading from file).
        """
        self.chat = chat
        self.deployment_name = deployment_name
        self.axiom_store = axiom_store

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

    def invoke(self, question: str) -> str:
        """
        Process a user question and generate a response using Azure OpenAI.

        This method:
        1. Loads and formats the constitution with axiom data
        2. Prepares the system prompt
        3. Formats the user prompt with the question and constitution
        4. Queries the Azure OpenAI model
        5. Returns the generated response

        Args:
            question: The user's question about health insurance.

        Returns:
            The AI-generated response based on the constitution and prompts.
        """

        # Load system prompt
        system_prompt = self._load_system_prompt()

        # Load and format user prompt
        user_prompt = self._load_and_format_user_prompt(question)

        # Create messages for the chat completion
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Query Azure OpenAI
        response = self.chat.chat.completions.create(
            model=self.deployment_name,
            messages=messages,  # type: ignore
            temperature=0.7,
            max_tokens=1000,
        )

        return response.choices[0].message.content or ""
