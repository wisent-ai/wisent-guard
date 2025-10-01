from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent_guard.core.contrastive_pairs.core.pair import ContrastivePair
from wisent_guard.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent_guard.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent_guard.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["CopaExtractor"]
_LOG = setup_logger(__name__)


class COPAExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the COPA benchmark."""

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from COPA docs.

        COPA schema:
            - premise: str
            - choice1, choice2: str
            - question: str
            - label: 0 or 1

        Args:
            lm_eval_task_data: lm-eval task instance for COPA.
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items)

        pairs: list[ContrastivePair] = []

        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid COPA pairs extracted", extra={"task": task_name})

        return pairs
    
    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single COPA doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            premise = str(doc.get("premise", "")).strip()
            choice1 = str(doc.get("choice1", "")).strip()
            choice2 = str(doc.get("choice2", "")).strip()
            question = str(doc.get("question", "")).strip()
            label = doc.get("label")


            if not premise or not choice1 or not choice2 or not question or not label in {0, 1}:
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"doc": doc},
                )
                return None
            
            fills = {"cause": "because", "effect": "therefore"}
            
            question = f"{premise.rstrip('.')} {fills[question]}"
            formatted_question = f"{question}\nA. {choice1}\nB. {choice2}"

            correct = choice1 if label == 0 else choice2
            incorrect = choice2 if label == 0 else choice1

            metadata = {
                "label": "copa",
            }

            return self._build_pair(
                question=formatted_question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:  
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

    @staticmethod
    def _build_pair(
        question: str,
        correct: str,
        incorrect: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContrastivePair:
        positive_response = PositiveResponse(model_response=correct)
        negative_response = NegativeResponse(model_response=incorrect)
        return ContrastivePair(prompt=question, positive_response=positive_response, negative_response=negative_response, label=metadata.get("label"))
