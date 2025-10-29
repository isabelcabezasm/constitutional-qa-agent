import argparse
import asyncio
from typing import Protocol 

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


async def run_evaluation(*, 
                         question_answer_fn: QuestionAnswerFunction,
                         input_data_path: str):
    """
    Run the evaluation process with the given data path.
    
    Args:
        data_path (str): Path to the data file or directory
    """
    print(f"Running evaluation with data path: {input_data_path}")
    # TODO: Implement evaluation logic here
    pass


