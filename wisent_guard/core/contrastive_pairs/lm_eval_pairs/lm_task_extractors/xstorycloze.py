from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent_guard.core.contrastive_pairs.core.pair import ContrastivePair
from wisent_guard.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent_guard.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent_guard.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["XStoryCloze"]
_LOG = setup_logger(__name__)


class XStoryClozeExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the XStoryCloze benchmark."""

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from XStoryCloze docs.

        XStoryCloze schema:
            - input_sentence_1, input_sentence_2, input_sentence_3, input_sentence_4: str
            - sentence_quiz1, sentence_quiz2: str
            - answer_right_ending: 1 or 2 or 3 or 4
            
        Args:
            lm_eval_task_data: lm-eval task instance for XStoryCloze.
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
            log.warning("No valid XStoryCloze pairs extracted", extra={"task": task_name})

        return pairs
    
    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single XStoryCloze doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            inputs = [str(doc.get("input_sentence_1", "")).strip(), str(doc.get("input_sentence_2", "")).strip(),
                      str(doc.get("input_sentence_3", "")).strip(), str(doc.get("input_sentence_4", "")).strip()] 
            endings = [str(doc.get("sentence_quiz1")).strip(), str(doc.get("sentence_quiz2")).strip()]
            answer = doc.get("answer_right_ending") - 1

            if not inputs or not endings or not answer:
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"doc": doc},
                )
                return None
            
            correct = endings[answer]
            incorrect = endings[(answer+1)%len(endings)]

            formatted_question = " ".join(s.strip() for s in inputs if s)
            formatted_question = f"{formatted_question}\n \nA. {incorrect}\nB. {correct}"

            metadata = {
                "label": "xstorycloze",
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