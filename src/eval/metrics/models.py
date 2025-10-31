from pydantic import BaseModel, Field


class Entity(BaseModel):
    """
    Represents a causal relationship between two variables.

    This class models an entity that captures the relationship between a trigger
    (cause) and its consequence (effect). It's typically used to represent
    cause-and-effect relationships in behavioral analysis, habit tracking, or
    outcome measurement scenarios.

    Attributes:
        trigger_variable (str): The name of the variable that acts as a trigger
            or cause, typically related to habits, activities, or input factors.
        consequence_variable (str): The name of the variable that represents
            the outcome or effect, typically related to results, effects, or
            output measures.
    Example: Tobacco use significantly increases mortality risk.
    """

    trigger_variable: str = Field(
        description="The name of the variable, related with habits, activities, ..."
    )
    consequence_variable: str = Field(
        description="The name of the variable, related with outcomes, effects, ..."
    )


class EntityExtraction(BaseModel):
    """
    A model for storing extracted entities from different sources in an evaluation
    context.
    """

    user_query_entities: list[str] = Field(
        description="Entities extracted from the user query."
    )
    llm_answer_entities: list[str] = Field(
        description="Entities extracted from the LLM answer."
    )
    expected_answer_entities: list[str] = Field(
        description="Entities extracted from the expected answer."
    )
