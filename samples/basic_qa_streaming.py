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


async def main():
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

    # Collect citations for reference section
    citations = []

    # Stream the response
    async for chunk in engine.invoke_streaming(question):
        match chunk:
            case TextContent():
                # Print text content as it arrives
                print(chunk.content, end="", flush=True)
            case CitationContent():
                # Print citation with visual styling
                axiom = chunk.axiom
                # Use ANSI color codes: cyan and bold for citations
                print(f"\033[1;36m[{axiom.id}]\033[0m", end="", flush=True)
                # Collect citation for reference section
                if axiom not in citations:
                    citations.append(axiom)

    print("\n" + "-" * 80)
    # Display references section if there are citations
    if citations:
        print("\n\033[1mReferences:\033[0m")
        print("=" * 80)
        for axiom in citations:
            print(f"\n\033[1;36m[{axiom.id}]\033[0m \033[1m{axiom.subject}\033[0m")
            print(f"  Entity: {axiom.entity}")
            print(f"  Trigger: {axiom.trigger}")
            print(f"  Conditions: {axiom.conditions}")
            print(f"  Description: {axiom.description}")
            print(f"  \033[2mCategory: {axiom.category}\033[0m")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
