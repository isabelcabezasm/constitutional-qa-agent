import argparse
import asyncio
from eval.eval import run_evaluation

async def generate_answer(*, query: str) -> str:
    print(f"Generating answer for query: {query}")
    return "Generated answer"


def main():
    """
    Main function that parses command line arguments and runs the evaluation.
    """
    parser = argparse.ArgumentParser(description="Run evaluation on provided data")
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to the data file or directory (mandatory)"
    )
    
    args = parser.parse_args()
    
    # Run the async evaluation function
    asyncio.run(run_evaluation(
        question_answer_fn=generate_answer,
        input_data_path=args.data_path
    ))


if __name__ == "__main__":
    main()
