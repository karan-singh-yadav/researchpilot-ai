from app.services.llm.base import LLMUsage


def estimate_cost_usd(usage: LLMUsage, input_per_mtok: float, output_per_mtok: float) -> float:
    """Estimate only: uses user-configured prices. Returns 0.0 when prices are not configured."""
    return (usage.input_tokens * input_per_mtok + usage.output_tokens * output_per_mtok) / 1_000_000
