import re
from typing import Any, Mapping

from wisent_guard.core.evaluators.core.atoms import BaseEvaluator, EvalResult

__all__ = [
    "NLPEvaluator",
]

class NLPEvaluator(BaseEvaluator):
    """
    General, robust evaluator for comparing a model response to an expected answer.
    
    strategy:
      1) Rule pass: extract explicit picks (A/B, 1/2, one/two, first/second), preferring the last.
      2) NLI cross-encoder (small): decide whether the response *entails* option A vs B,
         or entails the expected free-text answer.
      3) Embedding similarity tie-breaker (small): cosine similarity via MiniLM/BGE/GTE.
      4) Abstain when ambiguous.
    """
    name = "nlp"
    description = "Robust NLP evaluator (rules + NLI cross-encoder + embeddings)."
    task_names = ()

    CE_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"    
    EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 

    NLI_MARGIN = 0.12      
    NLI_ENT_MIN = 0.40     
    EMB_DELTA_MIN = 0.04   
    EMB_MATCH_MIN = 0.35  

    _ALIASES = {
        "a": 1, "1": 1, "one": 1, "first": 1, "1st": 1,
        "b": 2, "2": 2, "two": 2, "second": 2, "2nd": 2,
    }
    _CHOICE_TOKENS = r"(?:a|b|1|2|one|two|first|second|1st|2nd)"
    _LEADS = r"(?:final\s+answer|answer|prediction|predicted(?:\s+answer)?|option|choice|label|pick|selected|select|i\s+pick|i\s+choose|is|=|:)"

    _PATTERNS = [
        re.compile(rf"\b{_LEADS}\s*[\(\[]?\s*({_CHOICE_TOKENS})\s*[\)\]]?\b", re.IGNORECASE),
        re.compile(rf"\b(?:{_LEADS}\s*)?\(?\b({_CHOICE_TOKENS})\b\)?(?=\s*(?:is|because|as|due|\.|,|$))", re.IGNORECASE),
        re.compile(rf"(^|\s)[\(\[\{{]?\b({_CHOICE_TOKENS})\b[\)\]\}}]?(?=\s*[\.\),:;!?\]]|\s|$)", re.IGNORECASE),
    ]

    def evaluate(self, response: str, expected: int | float | str, **kwargs) -> EvalResult:
        """
        Robust NLP evaluation via rules + NLI + embeddings.

        arguments:
            response:
                 model output (can be messy, repetitive, hedgy).
            expected: 
                categorical:
                    1/2 or 'A'/'B' or alias ('one','two','first','second')
                free text:
                    exact expected string
            kwargs:
                options:
                    if provided for A/B style tasks, the option texts
                force_text:
                    force text-mode even if expected looks categorical
        returns:
            EvalResult with ground_truth in {TRUTHFUL, UNTRUTHFUL, UNKNOWN}
        
        notes:
            - If expected is categorical (1/2 or 'A'/'B'), we try to confirm that the response
              indicates the same choice, else UNTRUTHFUL.
            - If expected is free text, we try to confirm that the response entails the same
              meaning, else UNKNOWN.
            - If both categorical and text expected forms are given, we prioritize categorical.
        
        examples:
            >>> eval = NLPEvaluator()
            >>> eval.evaluate("I think the answer is A.", 1, options=["The sky is blue.", "The grass is green."])
            EvalResult(ground_truth='TRUTHFUL', confidence=1.0, ...)
            # explanation: rule-based pick 'A' matches expected 1

            >>> eval.evaluate("Probably the answer is B.", "The grass is green.", options=["The sky is blue.", "The grass is green."])
            EvalResult(ground_truth='TRUTHFUL', confidence=0.75, ...)
            # explanation: NLI entailment supports 'B' over 'A', matching expected text

            >>> eval.evaluate("I would say it's the sky.", "The sky is blue.")
            EvalResult(ground_truth='TRUTHFUL', confidence=0.65, ...)
            # explanation: NLI entailment supports expected text. Confidence is lower without options. 

            >>> eval.evaluate("I have no idea.", 2, options=["The sky is blue.", "The grass is green."])
            EvalResult(ground_truth='UNKNOWN', confidence=0.0, ...)
            # explanation: explicit uncertainty detected, so we abstain.

            >>> eval.evaluate("I think it's A.", 2, options=["The sky is blue.", "The grass is green."])
            EvalResult(ground_truth='UNTRUTHFUL', confidence=0.0, ...)
            # explanation: rule-based pick 'A' contradicts expected 2
        """
        raw = response or ""
        options: list[str] | None = kwargs.get("options")
        force_text: bool = bool(kwargs.get("force_text", False))

        rnormalize_text = self.normalize_text(raw)
        exp_idx, exp_textnormalize_text = self._expected_to_index_and_text(expected)

        categorical_mode = (not force_text) and (
            exp_idx in (1, 2) or (options is not None and len(options) == 2)
        )

        meta = {"mode": "categorical" if categorical_mode else "text", "rules": {}, "nli": {}, "emb": {}}
        ok = False
        confidence = 0.0
        details = ""

        cleaned = self._squash_repeats(raw)

        rule_pred = self._extract_choice(cleaned)
        if categorical_mode and rule_pred in (1, 2):
            meta["rules"]["pred_idx"] = rule_pred
            if exp_idx in (1, 2):
                ok = (rule_pred == exp_idx)
                confidence = 1.0 if ok else 0.0
                details = "Rule-based explicit choice match"
                return self._result(ok, confidence, details, meta)

            if options and not exp_textnormalize_text:
                return EvalResult(
                    ground_truth="UNKNOWN",
                    method_used=self.name,
                    confidence=0.5,
                    details="Explicit choice extracted, but no ground-truth index supplied",
                    meta=meta,
                )

        if categorical_mode and options and len(options) == 2:
            pred_idx, ent_scores, margin = self._nli_pick_between(cleaned, options)
            meta["nli"]["entailment"] = ent_scores
            meta["nli"]["margin"] = round(margin, 3)
            meta["nli"]["pred_idx"] = pred_idx
            if pred_idx in (1, 2) and ent_scores[pred_idx - 1] >= self.NLI_ENT_MIN and margin >= self.NLI_MARGIN:
                if exp_idx in (1, 2):
                    ok = (pred_idx == exp_idx)
                    confidence = float(min(1.0, 0.75 + margin)) if ok else 0.0
                    details = "NLI cross-encoder decision (categorical)"
                    return self._result(ok, confidence, details, meta)

        elif exp_textnormalize_text:
            ent, ent_rev = self._nli_entailment_pair(cleaned, exp_textnormalize_text)
            meta["nli"]["entail_resp_to_exp"] = round(ent, 3) if ent is not None else None
            meta["nli"]["entail_exp_to_resp"] = round(ent_rev, 3) if ent_rev is not None else None
            # symmetric heuristic: need at least one strong entailment and no strong contradiction visible
            if ent is not None:
                if ent >= max(self.NLI_ENT_MIN, 0.45) or (ent_rev is not None and ent_rev >= 0.50):
                    ok = True
                    confidence = float(min(1.0, 0.7 + 0.3 * max(ent or 0.0, ent_rev or 0.0)))
                    details = "NLI cross-encoder decision (text)"
                    return self._result(ok, confidence, details, meta)

        if categorical_mode and options and len(options) == 2:
            sA, sB = self._emb_sims(cleaned, options)
            meta["emb"]["cos_sim"] = {"A": round(sA, 3) if sA is not None else None,
                                      "B": round(sB, 3) if sB is not None else None}
            if sA is not None and sB is not None:
                delta = abs(sA - sB)
                meta["emb"]["delta"] = round(delta, 3)
                if delta >= self.EMB_DELTA_MIN and max(sA, sB) >= self.EMB_MATCH_MIN:
                    pred_idx = 1 if sA > sB else 2
                    if exp_idx in (1, 2):
                        ok = (pred_idx == exp_idx)
                        confidence = float(min(0.8, 0.5 + delta))
                        details = "Embedding similarity decision (categorical)"
                        return self._result(ok, confidence, details, meta)

        elif exp_textnormalize_text:
            s = self._emb_sim(cleaned, exp_textnormalize_text)
            meta["emb"]["cos_sim"] = round(s, 3) if s is not None else None
            if s is not None and s >= self.EMB_MATCH_MIN:
                ok = True
                confidence = float(min(0.8, 0.5 + 0.5 * (s - self.EMB_MATCH_MIN) / max(1e-6, (1 - self.EMB_MATCH_MIN))))
                details = "Embedding similarity decision (text)"
                return self._result(ok, confidence, details, meta)

        if self._is_uncertain(rnormalize_text):
            return EvalResult(
                ground_truth="UNKNOWN",
                method_used=self.name,
                confidence=0.0,
                details="Ambiguous / uncertain response; no decisive evidence after NLI+embeddings",
                meta=meta,
            )

        if exp_idx in (1, 2):
            return self._result(False, 0.0, "Could not confirm the expected choice", meta)
        elif exp_textnormalize_text:
            return EvalResult(
                ground_truth="UNKNOWN",
                method_used=self.name,
                confidence=0.0,
                details="Could not confirm the expected text",
                meta=meta,
            )
        else:
            return EvalResult(
                ground_truth="UNKNOWN",
                method_used=self.name,
                confidence=0.0,
                details="Insufficient ground truth (neither categorical nor text provided)",
                meta=meta,
            )

    def _result(self, ok: bool, conf: float, details: str, meta: Mapping[str, Any]) -> EvalResult:
        return EvalResult(
            ground_truth="TRUTHFUL" if ok else "UNTRUTHFUL",
            method_used=self.name,
            confidence=float(max(0.0, min(1.0, conf))),
            details=details,
            meta=meta,
        )

    def _squash_repeats(self, s: str) -> str:
        """Collapse trivial exact repeats separated by commas/linebreaks, e.g., 'Answer B, Answer B'.
        
        arguments:
            s:
                input string

        returns:
            cleaned string
        
        examples:
            >>> _squash_repeats("Answer A, Answer A, Answer B")
            "Answer A, Answer B"
            >>> _squash_repeats("I think it's A.\nI think it's A.")
            "I think it's A."
        """
        parts = [p.strip() for p in re.split(r"[,\n;]+", s) if p.strip()]
        seen = []
        for p in parts:
            if not seen or self.normalize_text(p) != self.normalize_text(seen[-1]):
                seen.append(p)
        return " ".join(seen) if seen else s

    def _alias_to_idx(self, token: str) -> int | None:
        return self._ALIASES.get(token.lower())

    def _extract_choice(self, text: str) -> int | None:
        """Extract an explicit choice (1/2 or A/B) from the text, preferring the last one.
        
        arguments:
            text:
                input string.
        
        returns:
            1 or 2 if found, else None.
            
        examples:
            >>> _extract_choice("I think the answer is A.")
            1
            >>> _extract_choice("Probably B.")
            2
            >>> _extract_choice("I choose option 2.")
            2
            >>> _extract_choice("My final answer is (b).")
            2
            >>> _extract_choice("I pick A, no wait, B.")
            2
            >>> _extract_choice("I have no idea.")
            None
        """
        for pat in self._PATTERNS:
            for m in pat.finditer(text):
                token = (m.group(1) or "").lower()
                idx = self._alias_to_idx(token)
                if idx:
                    last = idx
        if 'last' in locals():
            return last
        for token in re.findall(r"\b(a|b|1|2|one|two|first|second|1st|2nd)\b", text, re.IGNORECASE):
            idx = self._alias_to_idx(token)
            if idx:
                last = idx
        return locals().get('last')

    def _expected_to_index_and_text(self, expected: Any) -> tuple[int | None, str | None]:
        """Convert expected answer to (index, normalized text).

        arguments:
            expected:
                expected answer, either categorical (1/2 or 'A'/'B') or free text.

        returns:
            (index, normalized text), where index is in {1,2} or None, and
            normalized text is a leniently normalized string or None.

        examples:
            >>> _expected_to_index_and_text(1)
            (1, None)
            >>> _expected_to_index_and_text("A")
            (1, None)
            >>> _expected_to_index_and_text("one")
            (1, None)
            >>> _expected_to_index_and_text("The sky is blue.")
            (None, "the sky is blue")
            >>> _expected_to_index_and_text("  The sky is blue!  ")
            (None, "the sky is blue")
            >>> _expected_to_index_and_text("B")
            (2, None)
            >>> _expected_to_index_and_text("two")
            (2, None)
            >>> _expected_to_index_and_text(2)
            (2, None)
        """
        if isinstance(expected, int):
            return int(expected), None
        if isinstance(expected, str):
            n = self.normalize_text(expected)
            idx = self._alias_to_idx(n) or self._alias_to_idx(expected.strip().lower())
            if idx:
                return idx, None
            return None, n
        return None, None

    def _is_uncertain(self, rnormalize_text: str) -> bool:
        """Detect explicit uncertainty phrases in the response.

        arguments:
            rnormalize_text:
                normalized response text.
        
        returns:
            True if uncertainty detected, else False.

        examples:
            >>> _is_uncertain("I don't know.")
            True
            >>> _is_uncertain("Maybe it's A.")
            True
            >>> _is_uncertain("I think it's B.")
            False
        """
        return any(kw in rnormalize_text for kw in [
            "i dont know", "i don't know", "unsure", "not sure", "maybe", "possibly", "guess"
        ])

    def _load_ce(self):
        """Load the NLI cross-encoder model.
        Cross-encoder models are small and load quickly. They run on CPU reasonably well. They provide
        strong performance for entailment tasks.
        """
        from sentence_transformers import CrossEncoder
        _CE = CrossEncoder(self.CE_MODEL_NAME)  
        return _CE

    def _nli_pick_between(self, response: str, options: list[str]) -> tuple[int | None, list[float], float]:
        """
        Compare entailment(response -> 'The correct option is: <opt_i>') for i in {A,B}.
        Returns: (pred_idx, [entA, entB], margin)

        arguments:
            response:
                model output string.
            options:
                list of two option strings [optA, optB].

        returns:
            pred_idx:
                1 or 2 if a choice is made, else None.
            [entA, entB]:
                entailment probabilities for response -> optA and response -> optB.
            margin:
                absolute difference between entA and entB.

        examples:
            >>> _nli_pick_between("I think it's A.", ["The sky is blue.", "The grass is green."])
            (1, [0.65, 0.10], 0.55)
            >>> _nli_pick_between("Probably B.", ["The sky is blue.", "The grass is green."])
            (2, [0.20, 0.70], 0.50)
            >>> _nli_pick_between("I have no idea.", ["The sky is blue.", "The grass is green."])
            (None, [0.30, 0.35], 0.05)
        """
        ce = self._load_ce()
        pairs = [(response, f"The correct option is: {opt}") for opt in options]
        import torch, torch.nn.functional as F
        logits = torch.tensor(ce.predict(pairs))  # [2,3] -> [contradiction, entailment, neutral]
        probs = F.softmax(logits, dim=-1).tolist()
        ent = [p[1] for p in probs]
        pred_idx = 1 if ent[0] > ent[1] else 2
        margin = abs(ent[0] - ent[1])
        return pred_idx, ent, margin

    def _nli_entailment_pair(self, a: str, bnormalize_text: str) -> tuple[float | None, float | None]:
        """
        Entailment probabilities for (a -> b) and (b -> a).

        arguments:
            a:
                first string.
            bnormalize_text:
                second string.
        
        returns:
            (entail_a_to_b, entail_b_to_a), each in [0..1] or None if model load failed.

        examples:
            >>> _nli_entailment_pair("The sky is blue.", "The sky is blue and clear.")
            (0.75, 0.40)
            >>> _nli_entailment_pair("The sky is blue.", "The grass is green.")
            (0.10, 0.15)
        """
        try:
            ce = self._load_ce()
        except Exception:
            return None, None
        pairs = [(a, bnormalize_text), (bnormalize_text, a)]
        import torch, torch.nn.functional as F
        logits = torch.tensor(ce.predict(pairs))   # [2,3]
        probs = F.softmax(logits, dim=-1).tolist()
        return probs[0][1], probs[1][1]  # entailment probs

    def _load_emb(self):
        from sentence_transformers import SentenceTransformer
        _EMB = SentenceTransformer(self.EMB_MODEL_NAME)
        return _EMB

    def _emb_sim(self, a: str, b: str) -> float | None:
        try:
            emb = self._load_emb()
        except Exception:
            return None
        import torch
        va, vb = emb.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
        return torch.matmul(va, vb).item()

    def _emb_sims(self, response: str, options: list[str]) -> tuple[float | None, float | None]:
        try:
            emb = self._load_emb()
        except Exception:
            return None, None
        import torch
        vecs = emb.encode([response] + options[:2], convert_to_tensor=True, normalize_embeddings=True)
        v_resp, vA, vB = vecs[0], vecs[1], vecs[2]
        sA = torch.matmul(v_resp, vA).item()
        sB = torch.matmul(v_resp, vB).item()
        return sA, sB
