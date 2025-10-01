from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent_guard.core.contrastive_pairs.core.pair import ContrastivePair
from wisent_guard.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent_guard.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent_guard.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["SciQExtractor"]
_LOG = setup_logger(__name__)


class SciQExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the SciQ benchmark."""

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from SciQ docs.

        SciQ schema:
            - support: str
            - question: str
            - distractor1, distractor2, distractor3: str
            - correct_answer: str

        Args:
            lm_eval_task_data: lm-eval task instance for SciQ.
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
            log.warning("No valid SciQ pairs extracted", extra={"task": task_name})

        return pairs
    
    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single SciQ doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            support = str(doc.get("support", "")).strip()
            question = str(doc.get("question", "")).strip()
            correct = str(doc.get("correct_answer", "")).strip()
            incorrect = str(doc.get("distractor1", "")).strip() #take any distractor

            if not support or not question or not correct or not incorrect:
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"doc": doc},
                )
                return None
        
            formatted_question = f"{support}\nQuestion: {question}\nAnswer:\nA. {incorrect}\nB. {correct}"

            metadata = {
                "label": "sciq",
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