import argparse
import asyncio
from eval.eval import run_evaluation


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
    
    # TODO: For now, we'll need to provide a dummy question_answer_fn
    # This should be properly implemented based on your requirements
    async def dummy_qa_fn(*, query: str) -> str:
        return f"Generated answer for: {query}"
    
    # Run the async evaluation function
    asyncio.run(run_evaluation(
        question_answer_fn=dummy_qa_fn,
        input_data_path=args.data_path
    ))


if __name__ == "__main__":
    main()
