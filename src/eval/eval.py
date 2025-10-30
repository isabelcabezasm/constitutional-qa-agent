from datetime import datetime
from pathlib import Path
from typing import Protocol


def root() -> Path:
    """Returns the root directory of the project."""
    return Path(__file__).parents[2].resolve()


class QuestionAnswerFunction(Protocol):
    async def __call__(self, *, query: str) -> str:
        """
        Takes a user query and returns a generated scenario.

        Args:
            query: The user's query string

        Returns:
            A generated scenario as a string
        """
        ...


async def run_evaluation(
    *,
    question_answer_fn: QuestionAnswerFunction,
    input_data_path: str | None = None,
    ouptput_data_path: str | None = None,
) -> None:
    """
    Run the evaluation process with the given data path.

    Args:
        data_path (str): Path to the data file or directory
    """

    input_path = input_data_path or root() / "data/eval_dataset.json"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = ouptput_data_path or root() / f"runs/{timestamp}"
    print(f"Running evaluation with data path: {input_path}")
    print(f"Running evaluation with output data path: {output_path}")

    # TODO: Implement evaluation logic here
    pass
