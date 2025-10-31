This metric measures the accuracy of the predictions in the "generated answer" (e.g.,
quota or fee goes up / goes down, more healthy, less healthy) compared to the ground
truth for these entities: {{entity_list}} Evaluate the generated answer by assessing
whether the underlying semantics and behaviors of the predicted entities match those in
the expected answer, regardless of how they are specifically expressed. Determine
whether the entity's behavior in the generated answer aligns with that in the expected
answer. (more healthy, less healthy, quota behavior, etc) If it doesn't match or is
absent from the generated answer, assign an accuracy of 0.

This is the generated answer: {{llm_answer}} and this expected answer:
{{expected_answer}}