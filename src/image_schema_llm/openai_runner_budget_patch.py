"""
Patch notes for integrating RuntimeCostTracker into openai_runner.py.

This file is intentionally not imported by the application. It contains the
exact code pattern to apply inside run_openai_job().

Replace the manual cost calculation and append_jsonl(cost_log.jsonl, ...)
block with RuntimeCostTracker usage.
"""

# At top of src/image_schema_llm/openai_runner.py:
#
# from image_schema_llm.cost_tracker import RuntimeCostTracker
# from image_schema_llm.runtime_config import load_runtime_config

# Inside run_openai_job(), before creating the OpenAI client:
#
# runtime_config = load_runtime_config(project_root)
# tracker = RuntimeCostTracker.from_runtime_config(
#     project_root=project_root,
#     runtime_config=runtime_config,
# )
#
# if tracker.threshold_reached():
#     tracker.write_stop_record(
#         reason="spend_threshold_reached_before_call",
#         run_key=job.run_key,
#         metadata={
#             "provider": job.model.provider,
#             "model_id": job.model.model_id,
#             "prompt_id": job.prompt.prompt_id,
#             "condition_id": job.condition.condition_id,
#             "sentence_id": job.sentence.sentence_id,
#         },
#     )
#     manager.write_stop_record(
#         reason="spend_threshold_reached_before_call",
#         run_key=job.run_key,
#         details=tracker.summary_dict(),
#     )
#     return OpenAIRunResult(
#         run_key=job.run_key,
#         status="stopped",
#         message="Spend threshold reached before API call.",
#     )
#
# After response = client.generate(...):
#
# cost_record = tracker.record_api_usage(
#     run_key=job.run_key,
#     model=job.model,
#     input_tokens=response.input_tokens,
#     output_tokens=response.output_tokens,
#     provider=job.model.provider,
#     metadata={
#         "prompt_id": job.prompt.prompt_id,
#         "condition_id": job.condition.condition_id,
#         "sentence_id": job.sentence.sentence_id,
#     },
# )
#
# Use these fields in raw_record:
#
# "input_tokens": cost_record["input_tokens"],
# "output_tokens": cost_record["output_tokens"],
# "estimated_cost": cost_record["estimated_cost"],
# "currency": cost_record["currency"],
