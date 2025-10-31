Extract all entities from the following sentences.
Entities are defined as:

Physical activity types - such as aerobic, cardio, strength, mobility, sports, name of sports, etc..

Medical related behavior - like checkups, medical cost, quality of healthcare in the country where the client lives

Physical characteristics - age, gender, weight, etc..

Habits - like alcohol drinking, smoke, healthy nutrition, 

Ideally we want to have one entity that impact or affect the health or the quota of one customer, so if you find that structure, "link" them, adding them in the same object. 

If you find only a entity, without any related variable, just keep the string empty on that object.

Sentences:
user query: {{user_question}}
llm answer: {{llm_answer}}
expected answer: {{expected_answer}}
 