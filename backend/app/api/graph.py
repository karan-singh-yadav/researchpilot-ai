from fastapi import APIRouter, Depends

from app.graph.builder import graph_mermaid
from app.security.auth import require_api_key
from app.security.rate_limit import rate_limit

router = APIRouter(
    prefix="/api/graph",
    tags=["graph"],
    dependencies=[Depends(rate_limit), Depends(require_api_key)],
)


@router.get("")
def get_graph() -> dict:
    """The actual compiled LangGraph topology as a Mermaid diagram."""
    return {"mermaid": graph_mermaid()}
