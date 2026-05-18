_PROMPT = """
Please translate the following text from {source_language} to {target_language}.
You MUST return ONLY valid JSON. Do not include markdown, code fences, or explanations.
JSON schema:
{
"llm_oracle": string|null
}
If you do not receive the WAV file, enter llm_oracle as 'n/a'.
Do NOT mention that you need the WAV file, only give the JSON schema output.
If you violate this, the output will be discarded.
"""