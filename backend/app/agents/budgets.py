from app.schemas.plan import ExecutionBudget

DEPTH_BUDGETS: dict[str, ExecutionBudget] = {
    "quick": ExecutionBudget(
        max_subtasks=3, max_sources=5, max_iterations=1, max_revisions=1, timeout_seconds=120
    ),
    "standard": ExecutionBudget(
        max_subtasks=6, max_sources=12, max_iterations=2, max_revisions=2, timeout_seconds=300
    ),
    "deep": ExecutionBudget(
        max_subtasks=10, max_sources=25, max_iterations=3, max_revisions=3, timeout_seconds=900
    ),
}


def budget_for(depth: str) -> ExecutionBudget:
    try:
        return DEPTH_BUDGETS[depth].model_copy()
    except KeyError:
        raise ValueError(f"unknown research depth '{depth}'") from None
