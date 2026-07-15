# Shared output-token cap for all providers. Generous because reasoning
# ("thinking") models spend most of their budget on chain-of-thought before
# writing the answer — a small cap gets exhausted mid-thought and the summary
# never arrives. Non-reasoning models stop early and don't use the headroom.
MAX_OUTPUT_TOKENS = 8192
