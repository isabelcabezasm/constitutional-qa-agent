"""
Simple main script for testing the QA Engine.

This script demonstrates basic usage of the QA Engine with a sample question.
"""

from core.dependencies import qa_engine


def main():
    """Run a simple test of the QA Engine."""
    # Initialize the QA engine
    engine = qa_engine()

    # Test question
    test_question = "What happens if I miss my premium payment?"

    print("=" * 80)
    print("QA Engine Test")
    print("=" * 80)
    print(f"\nQuestion: {test_question}\n")
    print("Processing...\n")

    # Get response
    response = engine.invoke(test_question)

    print("Response:")
    print("-" * 80)
    print(response)
    print("-" * 80)
    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()
