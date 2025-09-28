from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch


from wisent_guard.core.contrastive_pairs.core.pair import ContrastivePair
from wisent_guard.core.activations.core.atoms import LayerActivations, ActivationAggregationStrategy, LayerName, RawActivationMap
from wisent_guard.core.models.wisent_model import WisentModel 
__all__ = ["ActivationCollector"]

@dataclass(slots=True)
class ActivationCollector:
    """
        Collect per-layer activations for (prompt + response) using a chat template.

        arguments:
            model:
                :class: WisentModel
            store_device:
                Device to store collected activations on (default "cpu").
            dtype:
                Optional torch.dtype to cast activations to (e.g., torch.float32).
                If None, keep original dtype.

        detailed explanation:

        Let:
        - L = 4 transformer blocks
        - hidden size H = 256
        - prompt tokenized length T_prompt = 14
        - full sequence (prompt + response) tokenized length T_full = 22

        Step 1: Build templated strings (NOT tokenized yet)
            prompt_text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True
            )
            full_text   = tok.apply_chat_template(
                [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}],
                tokenize=False, add_generation_prompt=False
            )

        Step 2: Tokenize both with identical flags
            prompt_enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
            full_enc   = tok(full_text,   return_tensors="pt", add_special_tokens=False)

        Shapes:
            prompt_enc["input_ids"].shape == (1, T_prompt) == (1, 14)
            full_enc["input_ids"].shape   == (1, T_full)   == (1, 22)

        Boundary:
            prompt_len = prompt_enc["input_ids"].shape[-1] == 14
            continuation tokens in the full sequence start at index 14.

        Step 3: Forward pass with hidden states
            out = model.hf_model(**full_enc, output_hidden_states=True, use_cache=False)
            hs  = out.hidden_states

        hs is a tuple of length L + 1 (includes embedding layer at index 0):
            len(hs) == 5  -> indices: 0=embeddings, 1..4 = blocks
            Each hs[i].shape == (1, T_full, H) == (1, 22, 256)

        We map layer names "1".."L" to hs[1]..hs[L]:
            "1" -> hs[1], "2" -> hs[2], ..., "4" -> hs[4]

        Step 4: Per-layer extraction
            For a chosen layer i (1-based), get hs[i].squeeze(0) -> shape (T_full, H) == (22, 256)

            If return_full_sequence=True:
                store value with shape (T_full, H) == (22, 256)
            Else (aggregate to a single vector [H]):
                - CONTINUATION_TOKEN / CHOICE_TOKEN: take first continuation token -> cont[0] -> (H,)
                - FIRST_TOKEN:     layer_seq[0]    -> (H,)
                - LAST_TOKEN:      layer_seq[-1]   -> (H,)
                - MEAN_POOLING:    cont.mean(0)    -> (H,)
                - MAX_POOLING:     cont.max(0)[0]  -> (H,)

            where:
                layer_seq = hs[i].squeeze(0)                # (22, 256)
                cont_start = prompt_len = 14
                cont = layer_seq[14:]                       # (22-14=8, 256)

        Step 5: Storage and return
            - We move each stored tensor to 'store_device' (default "cpu") and cast to 'dtype'
            if provided (e.g., float32).
            - Keys are layer names: "1", "2", ..., "L".
            - Results are wrapped into LayerActivations with `activation_aggregation_strategy`
            set to your chosen strategy (or None if keeping full sequences).

        examples:
            Example usage (aggregated vectors per layer)
                >>> collector = ActivationCollector(model=my_wrapper, store_device="cpu", dtype=torch.float32)
                >>> updated_pair = collector.collect_for_pair(
                ...     pair,
                ...     layers=["1", "3"],  # subset (or None for all)
                ...     aggregation=ActivationAggregationStrategy.CONTINUATION_TOKEN,
                ...     return_full_sequence=False,
                ... )
                >>> pos_acts = updated_pair.positive_response.layers_activations
                >>> pos_acts.summary()
                    {
                    '1': {'shape': (256,), 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False},
                    '3': {'shape': (256,), 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False},
                    '_activation_aggregation_strategy': {'strategy': 'continuation_token'}
                    }

            Example usage (full sequences per layer)
                >>> updated_pair = collector.collect_for_pair(
                ...     pair,
                ...     layers=None,  # all layers "1".."L"
                ...     aggregation=ActivationAggregationStrategy.MEAN_POOLING,  # ignored when return_full_sequence=True
                ...     return_full_sequence=True,
                ... )
                >>> neg_acts = updated_pair.negative_response.layers_activations
                >>> # Suppose L=4 and T_full=22, H=256
                >>> neg_acts.summary()
                    {
                    '1': {'shape': (22, 256), 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False},
                    '2': {'shape': (22, 256), 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False},
                    '3': {'shape': (22, 256), 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False},
                    '4': {'shape': (22, 256), 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False},
                    '_activation_aggregation_strategy': {'strategy': None}
                    }
    """

    model: WisentModel
    store_device: str | torch.device = "cpu"
    dtype: torch.dtype | None = None

    def collect_for_pair(
        self,
        pair: ContrastivePair,
        layers: Sequence[LayerName] | None = None,  
        aggregation: ActivationAggregationStrategy = ActivationAggregationStrategy.CONTINUATION_TOKEN,
        return_full_sequence: bool = False,
        normalize_layers: bool = False,
    ) -> ContrastivePair:
        pos = self._collect_for_texts(pair.prompt, _resp_text(pair.positive_response),
                                      layers, aggregation, return_full_sequence, normalize_layers)
        neg = self._collect_for_texts(pair.prompt, _resp_text(pair.negative_response),
                                      layers, aggregation, return_full_sequence, normalize_layers)
        return pair.with_activations(positive=pos, negative=neg)

    def _collect_for_texts(
        self,
        prompt: str,
        response: str,
        layers: Sequence[LayerName] | None,
        aggregation: ActivationAggregationStrategy,
        return_full_sequence: bool,
        normalize_layers: bool = False,
    ) -> LayerActivations:
        
        self._ensure_eval_mode()
        with torch.inference_mode():
            tok = self.model.tokenizer # type: ignore[union-attr]
            if not hasattr(tok, "apply_chat_template"):
                raise RuntimeError("Tokenizer has no apply_chat_template; set it up or use a non-chat path.")

            # 1) Build templated strings
            prompt_text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tok.apply_chat_template(
                [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}],
                tokenize=False,
                add_generation_prompt=False,
            )

            # 2) Tokenize both with identical flags
            prompt_enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
            full_enc   = tok(full_text,   return_tensors="pt", add_special_tokens=False)

            # 3) Boundary from prompt-only tokens (CPU is fine)
            prompt_len = int(prompt_enc["input_ids"].shape[-1])

            # 4) Move only the batch that goes into the model
            compute_device = getattr(self.model, "compute_device", None) or next(self.model.hf_model.parameters()).device
            full_enc = {k: v.to(compute_device) for k, v in full_enc.items()}

            # 5) Forward on the full sequence to get hidden states
            out = self.model.hf_model(**full_enc, output_hidden_states=True, use_cache=False)
            hs: tuple[torch.Tensor, ...] = out.hidden_states  # hs[0]=emb, hs[1:]=layers

            if not hs:
                raise RuntimeError("No hidden_states returned. Can be due to model not supporting it.")

            n_blocks = len(hs) - 1
            names_by_idx = [str(i) for i in range(1, n_blocks + 1)]

            keep = self._select_indices(layers, n_blocks)
            collected: RawActivationMap = {}

            for idx in keep:
                name = names_by_idx[idx]
                h = hs[idx + 1].squeeze(0)  # [1, T, H] -> [T, H]
                if return_full_sequence:
                    value = h
                else:
                    value = self._aggregate(h, aggregation, prompt_len)
                value = value.to(self.store_device)
                if self.dtype is not None:
                    value = value.to(self.dtype)
                
                if normalize_layers:
                    value = self._normalization(value)

                collected[name] = value

            return LayerActivations(
                collected,
                activation_aggregation_strategy=None if return_full_sequence else aggregation,
            )

    def _select_indices(self, layer_names: Sequence[str] | None, n_blocks: int) -> list[int]:
        """Map layer names '1'..'L' -> indices 0..L-1."""
        if not layer_names:
            return list(range(n_blocks))
        out: list[int] = []
        for name in layer_names:
            try:
                i = int(name)
            except ValueError:
                raise KeyError(f"Layer name must be numeric string like '3', got {name!r}")
            if not (1 <= i <= n_blocks):
                raise IndexError(f"Layer '{i}' out of range 1..{n_blocks}")
            out.append(i - 1)
        return sorted(set(out))

    def _aggregate(
        self,
        layer_seq: torch.Tensor,  # [T, H]
        aggregation: ActivationAggregationStrategy,
        prompt_len: int,
    ) -> torch.Tensor:          # [H]
        if layer_seq.ndim != 2:
            raise ValueError(f"Expected [seq_len, hidden_dim], got {tuple(layer_seq.shape)}")

        # continuation = tokens after the prompt boundary
        cont_start = min(max(prompt_len, 0), layer_seq.shape[0] - 1)
        cont = layer_seq[cont_start:] if cont_start < layer_seq.shape[0] else layer_seq[-1:].contiguous()
        if cont.numel() == 0:
            cont = layer_seq[-1:].contiguous()

        s = aggregation
        if s in (ActivationAggregationStrategy.CHOICE_TOKEN,
                 ActivationAggregationStrategy.CONTINUATION_TOKEN):
            return cont[0]
        if s is ActivationAggregationStrategy.FIRST_TOKEN:
            return layer_seq[0]
        if s is ActivationAggregationStrategy.LAST_TOKEN:
            return layer_seq[-1]
        if s is ActivationAggregationStrategy.MEAN_POOLING:
            return cont.mean(dim=0)
        if s is ActivationAggregationStrategy.MAX_POOLING:
            return cont.max(dim=0).values
        return cont[0]

    def _normalization(
        self,
        x: torch.Tensor,
        dim: int = -1,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        Safely L2-normalize 'x' along 'dim'.

        arguments:
            x:
                Tensor of the shape [..., H] or [T, H]
            dim:
                Dimension along which to normalize (default -1, the last dimension).
            eps:
                Small value to avoid division by zero (default 1e-12).

        returns:
            L2-normalized tensor of the same shape as 'x'.
        """
        if not torch.is_floating_point(x):
            return x
        
        norm = torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=True)

        mask = norm > eps

        safe_norm = torch.where(mask, norm, torch.ones_like(norm))
        y = x / safe_norm
        y = torch.where(mask, y, torch.zeros_like(y))

        return y

    def _ensure_eval_mode(self) -> None:
        try:
            self.model.hf_model.eval()
        except Exception:
            pass

def _resp_text(resp_obj: object) -> str:
    for attr in ("model_response", "text"):
        if hasattr(resp_obj, attr) and isinstance(getattr(resp_obj, attr), str):
            return getattr(resp_obj, attr)
    return str(resp_obj)

if __name__ == "__main__":
    from wisent_guard.core.contrastive_pairs.core.pair import ContrastivePair
    from wisent_guard.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse

    model = WisentModel(name="/home/gg/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6")
    collector = ActivationCollector(model=model, store_device="cpu")

    pair = ContrastivePair(
        prompt="The capital of France is",
        positive_response=PositiveResponse(" Paris."),
        negative_response=NegativeResponse(" London."),
    )

    updated = collector.collect_for_pair(
        pair,
        layers=["1", "3"],
        aggregation=ActivationAggregationStrategy.CONTINUATION_TOKEN,
        return_full_sequence=False,
    )

    print(updated)

