from typing import Literal, TypeVar

from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient
from anyio import Path
from pydantic import BaseModel

from eval.metrics.models import EntityExtraction

T = TypeVar("T", bound=BaseModel)


class QAEvalEngine:
    """
    Question-Answering engine for health insurance queries.

    This class handles the orchestration of prompts, constitution loading,
    and interaction with Azure OpenAI via the Microsoft Agent Framework to provide
    contextualized responses based on predefined axioms.

    Attributes:
        agent (ChatAgent): The chat agent for model inference.
        axiom_store (AxiomStore | None): Storage for axioms/constitution data.
    """

    _INPUT_VARIABLES = [
        "entity_list",
        "llm_answer",
        "user_query",
        "expected_answer",
        "expected_entities",
        "generated_entities",
    ]

    PromptTypes = Literal["accuracy", "entity_extraction", "system", "topic_coverage"]

    def __init__(
        self,
        chat: AzureOpenAIChatClient,
    ):
        """
        Initialize the QA Engine.

        Args:
            chat: Azure OpenAI chat client instance for model inference.
            axiom_store: Optional storage for axioms (defaults to loading from file).
        """
        self.chat = chat
        self.agent: ChatAgent | None = None

    def _get_prompt(self, promptType: PromptTypes) -> str:
        """Load prompts."""

        file_path = Path(__file__).parent / "prompts" / f"{promptType}_prompt.md"
        with open(file_path, encoding="utf-8") as f:
            return f.read()

    async def _perform_model_invocation(self, prompt: str, output_type: type[T]) -> T:
        """Invoke the model and parse the output into the specified Pydantic model."""

        # Load system prompt
        system_prompt = self._get_prompt("system")

        # Create agent with system instructions if not already created
        # or if we need to update instructions
        if self.agent is None:
            self.agent = self.chat.create_agent(instructions=system_prompt)

        # Use asyncio to run the async agent
        response = await self.agent.run(prompt, response_format=output_type)
        assert isinstance(response.value, output_type)
        return response.value

    async def entity_extraction(
        self, user_query: str, llm_answer: str, expected_answer: str
    ) -> EntityExtraction:
        """
        Extract and compare entities between the LLM-generated answer and expected
        answer.
        """
        metric_prompt = self._get_prompt("entity_extraction").format(
            user_query=user_query,
            llm_answer=llm_answer,
            expected_answer=expected_answer,
        )
        return await self._perform_model_invocation(metric_prompt, EntityExtraction)

    async def accuracy_evaluator(self):
        """
        Evaluates the accuracy of question-answering responses.
        """
        pass

    async def topic_coverage_evaluator(self):
        """
        Evaluate the topic coverage of the LLM-generated answer against the expected
        answer.
        """
        pass
