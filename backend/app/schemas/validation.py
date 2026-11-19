from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DISCLAIMER = (
    "Automated checks. Citation IDs, statistics and quotations are verified deterministically "
    "against collected evidence; the claim-support review is performed by an LLM and "
    "can be wrong. Passing does not guarantee factual correctness."
)


class ValidationIssue(BaseModel):
    severity: Literal["error", "warning"]
    code: str
    message: str
    section: str | None = None
    method: Literal["deterministic", "llm"] = "deterministic"


class ValidationReport(BaseModel):
    passed: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    claims_checked: int = 0
    llm_judged: bool = False
    disclaimer: str = DISCLAIMER

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]


class ClaimVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    claim_id: int
    verdict: Literal["supported", "partial", "unsupported"]
    reason: str = Field(default="", max_length=400)


class CriticVerdictDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")
    verdicts: list[ClaimVerdict] = Field(min_length=1, max_length=40)
