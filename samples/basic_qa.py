"""
Example demonstrating how to use the QA Engine streaming API.

This example shows how to:
1. Use the streaming API to get real-time responses
2. Handle both text content and citation content
3. Process citations to display axiom information
"""

import asyncio

from core.dependencies import qa_engine
from core.qa_engine import CitationContent, TextContent


async def streaming_example():
    """Demonstrate streaming response with citation handling."""
    # Initialize the QA engine
    engine = qa_engine()

    # Example question
    question = (
        "What if I start exercising regularly and lose 20 pounds - "
        "how would that affect my future premium and coverage?"
    )

    print("Streaming Response Example")
    print("=" * 80)
    print(f"Question: {question}\n")
    print("Response:")
    print("-" * 80)

    # Stream the response
    async for chunk in engine.invoke_streaming(question):
        match chunk:
            case TextContent():
                # Print text content as it arrives
                print(chunk.content, end="", flush=True)
            case CitationContent():
                # Print citation with axiom information
                axiom = chunk.axiom
                print(f"[{axiom.id}]", end="", flush=True)

                # Optionally, you can collect citations for a reference section
                # citations.append(axiom)

    print("\n" + "-" * 80)


async def non_streaming_example():
    """Demonstrate non-streaming (complete) response."""
    # Initialize the QA engine
    engine = qa_engine()

    # Example question
    question = (
        "What if I quit smoking today - what would be the long-term "
        "impact on my health insurance costs and claim patterns?"
    )

    print("\nNon-Streaming Response Example")
    print("=" * 80)
    print(f"Question: {question}\n")
    print("Response:")
    print("-" * 80)

    # Get complete response
    response = await engine.invoke(question)
    print(response)
    print("-" * 80)


async def main():
    """Run both examples."""
    await streaming_example()
    await non_streaming_example()


if __name__ == "__main__":
    asyncio.run(main())
