
# Tutor-facing summary

The project is not designed to prove that LLMs possess embodied cognition. Instead, it evaluates whether image-schema prompting provides a useful intermediate representational layer beyond ordinary paraphrase.

The current analysis separates ordinary free-text interpretation from structured semantic annotation. Naïve prompt outputs are expected to appear as `free_text_unparsed` because they are preserved as free-text baselines rather than parsed as JSON. Direct-schema and structured role-based prompts are the structured-output conditions used for schema accuracy and literal/metaphorical/control classification.

The key methodological value is that image-schema prompting produces explicit and comparable fields: predicted schema, literal/metaphorical/control label, schematic roles, source/target domains, parse quality, and confidence. These can be scored against a human-validated sentence corpus.

The results should therefore be interpreted as evidence about structured semantic recoverability and prompt behaviour, not as evidence that models have human-like embodied experience. Strong performance on literal and metaphorical spatial sentences suggests that models can recover image-schema-like patterns when cues are clear. Weaker performance on weak-schema controls would indicate over-application and the need for better abstention mechanisms.


## GPT-assisted interpretation notes

### Accuracy by sentence type
- Schema accuracy is highest for **metaphorical spatial** sentences (0.9058), followed by **literal spatial** sentences (0.8088), and is much lower for **weak-schema controls** (0.5794). This suggests image-schema prompts align especially well with spatial and metaphorical-spatial material.

- Literal/metaphorical classification accuracy is near ceiling for **literal spatial** (0.9925) and **metaphorical spatial** (0.9992), but much weaker for **control weak-schema** items (0.3098), consistent with controls being less naturally captured by the literal/metaphorical spatial distinction.

- The differing `n_schema` and `n_lm` values indicate that metrics were computed only where the relevant usability flags permitted inclusion; partial records can still validly contribute to one metric if usable for that metric.

- Methodologically, naive free-text outputs should not be treated as parse failures: `free_text_unparsed` is expected for the naive baseline. JSON parsing quality should be assessed mainly for the structured prompt families.

Overall, the results indicate that structured image-schema interpretation is more successful for spatial and metaphorical-spatial sentences than for weak-schema controls. This supports the usefulness of image-schema prompting as an intermediate representational layer, but it should not be taken as evidence that the model possesses embodied cognition.

### Prompt family by sentence type
- For literal and metaphorical spatial sentences, both structured prompt families perform strongly: literal/metaphorical accuracy is near ceiling, and schema accuracy is high, especially for metaphorical spatial items.

- Direct-schema prompting gives slightly higher schema accuracy than structured-role prompting for literal and metaphorical cases, but performs very poorly on control weak-schema literal/metaphorical classification.

- Structured-role prompting is more robust on control weak-schema items, with higher schema accuracy and much higher literal/metaphorical/control accuracy, suggesting better handling of cases where no strong spatial schema should be imposed.

- The differing `n_schema` and `n_lm` values indicate that usability flags matter: records can validly contribute to one metric but not the other, so schema accuracy and literal/metaphorical accuracy should be interpreted separately.

- Methodologically, JSON/parse quality should be assessed mainly for the structured prompt families; naive free-text outputs, if present elsewhere as `free_text_unparsed`, are expected baseline behaviour rather than parse failures.

Overall, the results are consistent with the claim that structured image-schema prompting can provide a useful intermediate representational layer, especially for spatial and metaphorical-spatial interpretation. However, they do not show that LLMs possess embodied cognition, and the weak-schema controls caution against overinterpreting schema assignment as semantic understanding.
