from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from wisent_guard.core.models.core.atoms import SteeringPlan, SteeringVector, HookHandleGroup, GenerationStats, TopLogits
from wisent_guard.core.activations.core.atoms import RawActivationMap

from wisent_guard.core.prompts.core.atom import ChatMessage
from wisent_guard.core.utils.device import resolve_default_device, resolve_torch_device
from wisent_guard.core.contrastive_pairs.diagnostics import run_control_vector_diagnostics

_all__ = ["WisentModel"]


logger = logging.getLogger(__name__)

class WisentModel:
    """
    Wrapper around a causal LM (HF transformers) with steering capabilities.

    atributes:
        model_name:
            HF repo id or local path (e.g., 'meta-llama/Llama-3-8B-Instruct', 'Qwen/Qwen2.5-7B-Instruct).
        device:
            'cuda', 'cuda:0', 'cpu', etc. If None, leave to HF defaults/accelerate.
        hf_model:
            the loaded PreTrainedModel instance.    
        tokenizer:
            the loaded PreTrainedTokenizerBase instance.
        hidden_size:
            model hidden size (last dim of residual stream).
        num_layers:
            number of decoder blocks we can hook.
        _steering_plan:
            current SteeringPlan (can be empty).
        _hook_group:
            manages active hooks for clean detach.
    """
    def __init__(self, model_name: str, layers: RawActivationMap, device: str | None = None, hf_model: AutoModelForCausalLM | None = None):
        """
        Initialize the wrapper (model + tokenizer + default steering plan).

        arguments:
            model_name:
                HF repo id or local path (e.g., 'meta-llama/Llama-3-8B-Instruct', 'Qwen/Qwen2.5-7B-Instruct').
            layers:
                RawActivationMap of steering vectors (layer_name -> tensor), optional (can be {}).
            device:
                'cuda', 'cuda:0', 'cpu', etc. If None, leave to HF defaults/accelerate.
            hf_model:
                optional preloaded model (skips from_pretrained if provided).


        example:
            >>> wm = WisentModel("meta-llama/Meta-Llama-3-8B-Instruct", layers={}, device="cuda")
            >>> # Later, set steering vectors for layers "6" and "12":
            >>> wm.set_steering_from_raw({"6": torch.randn(wm.hidden_size), "12": torch.randn(wm.hidden_size)}, scale=0.7)
            >>> text = wm.generate("Say hi in 5 words.", max_new_tokens=10, use_steering=True)[0]
        """
        self.model_name = model_name
        self.device = device or resolve_default_device()

        self.hf_model: PreTrainedModel = hf_model or AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map=None,            
            trust_remote_code=True,   
        )
        self.hf_model.to(self.device)

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, trust_remote_code=True
        )

        if not self._is_chat_tokenizer():
            raise ValueError("Tokenizer does not support chat templates (missing apply_chat_template method). Change to a chat-capable model.")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.hf_model.generation_config, "pad_token_id", None) is None:
            self.hf_model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self._steering_plan: SteeringPlan = SteeringPlan.from_raw(layers or {})
        self._hook_group = HookHandleGroup()

        self._layers, self._hidden_size = self._resolve_decoder_layers_and_hidden()


    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    def _resolve_decoder_layers_and_hidden(self) -> tuple[list[nn.Module], int]:
        m = self.hf_model
        hidden_size = getattr(m.config, "hidden_size", None) or getattr(m.config, "n_embd", None)
        layers: list[nn.Module] = []

        # Most common homes for decoder blocks
        candidates = [
            "layers",                 
            "model.layers",           
            "model.decoder.layers",   
            "transformer.h",          
            "base_model.model.layers",
            "blocks", "model.blocks", 
        ]
        for path in candidates:
            obj = m
            try:
                for attr in path.split("."):
                    if attr:
                        obj = getattr(obj, attr)
                if (isinstance(obj, nn.ModuleList) or
                    (isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], nn.Module))):
                    layers = list(obj)
                    break
            except AttributeError:
                continue

        if not layers:
            raise RuntimeError("Could not resolve decoder layers for steering hooks.")

        if hidden_size is None:
            for p in m.parameters():
                if p.ndim >= 2:
                    hidden_size = int(p.shape[-1]); break
        if hidden_size is None:
            raise RuntimeError("Could not infer hidden size from model config.")

        return layers, int(hidden_size)

    def _is_chat_tokenizer(self) -> bool:
        return hasattr(self.tokenizer, "apply_chat_template") and callable(getattr(self.tokenizer, "apply_chat_template"))

    def apply_steering(self, plan: SteeringPlan | None = None) -> None:
        """
        Register forward hooks to add steering vectors *after* the selected decoder blocks.
        If plan is None, use the internal plan set at init or via set_steering_from_raw().
        Multiple vectors per layer are summed inside the hook.

        arguments:
            plan:
                optional SteeringPlan to use for this call only (overrides internal plan).
                If None, uses the internal plan.

        example:
            >>> wm.apply_steering()   # uses current internal plan
            >>> # ... generate ...
            >>> wm.detach()           # back to vanilla
        """
        p = plan or self._steering_plan
        if p.is_empty():
            return

        p.validate_hidden_size(hidden_size=self._hidden_size)
        self.detach() 

        name_to_index = {str(i + 1): i for i in range(len(self._layers))}

        for lname, vecs in p.layers.items():
            if lname not in name_to_index:
                continue
            idx = name_to_index[lname]
            layer = self._layers[idx]

            def _hook_factory(vlist: list[SteeringVector]):
                def _hook(_mod: nn.Module, _inp: tuple, out: torch.Tensor | tuple) -> torch.Tensor | tuple:
                    if isinstance(out, tuple):
                        hs = out[0]
                        delta = torch.zeros_like(hs)
                        for sv in vlist:
                            delta = delta + sv.materialize(hs)
                        return (hs + delta,) + out[1:]
                    else:
                        hs = out
                        delta = torch.zeros_like(hs)
                        for sv in vlist:
                            delta = delta + sv.materialize(hs)
                        return hs + delta
                return _hook

            handle = layer.register_forward_hook(_hook_factory(vecs))
            self._hook_group.add(handle)

    def detach(self) -> None:
        """
        Remove all registered steering hooks; model returns to unsteered behavior.
        """
        self._hook_group.remove_all()

    @contextmanager
    def detached(self):
        """
        Context manager that guarantees a vanilla (unsteered) model inside the block.

        example:
            >>> with wm.detached():
            ...     txt = wm.generate([[{"role": "user", "content": "Plain run"}]], use_steering=False)[0]
        """
        self.detach()
        try:
            yield
        finally:
            self.detach()

    def _encode_one(
        self,
        message: list[ChatMessage],
        add_generation_prompt: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Encode a single input in chat format.

        arguments:
            messages:
                list of {'role': str, 'content': str} dicts (chat messages).
            add_generation_prompt:
                If True, append the model's generation prompt at the end.

        returns:
            dict with 'input_ids' and 'attention_mask' tensors.

        example:
            >>> msgs = [
            ...   {"role":"system","content":"Be concise."},
            ...   {"role":"user","content":"Two bullet points about koalas."}
            ... ]
            >>> wm._encode_one(msgs, add_generation_prompt=True)
            {"input_ids": tensor([[...]]), "attention_mask": tensor([[...]])}
        """
      
        ids = self.tokenizer.apply_chat_template(
            message, tokenize=True, add_generation_prompt=add_generation_prompt, return_tensors="pt"
        )[0] 
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
        }

    def _batch_encode(
        self,
        inputs: list[list[ChatMessage]],
        add_generation_prompt: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Batch-encode a list of chat messages.

        arguments:
            inputs:
                list of chat messages (each a list of {'role','content'} dicts).
            add_generation_prompt:
                If True, append the model's generation prompt at the end of each.
        
        returns:
            dict with batched 'input_ids' and 'attention_mask' tensors.

        example:
            >>> msgs1 = [
            ...   {"role":"system","content":"Be concise."},
            ...   {"role":"user","content":"Two bullet points about koalas."}
            ... ]
            >>> msgs2 = [
            ...   {"role":"user","content":"Write a haiku about rain."}
            ... ]
            >>> wm._batch_encode([msgs1, msgs2], add_generation_prompt=True)
            {"input_ids": tensor([[...],[...]]), "attention_mask": tensor([[...],[...]])}
        """
        
        singles = []
        for item in inputs:
            singles.append(self._encode_one(item, add_generation_prompt=add_generation_prompt))

        batch = self.tokenizer.pad(singles, padding=True, return_tensors="pt")

        batch = {k: v.to(resolve_torch_device()) for k, v in batch.items()}

        return batch

    @torch.inference_mode()
    def generate(
        self,
        inputs: list[list[ChatMessage]],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        use_steering: bool = True,
        steering_plan: SteeringPlan | None = None,
        **gen_kwargs: Any,
    ) -> list[str]:
        """
        Batched text generation with optional steering.

        attributes:
            inputs:
                list of chat messages (each a list of {'role','content'} dicts).
            max_new_tokens:
                max tokens to generate (beyond the prompt).
            temperature:
                sampling temperature (0 = greedy, 1 = default sampling).
            top_p:
                nucleus sampling probability (1.0 = no nucleus).
            do_sample:
                if False, uses greedy decoding (top_k=1).
            num_return_sequences:
                number of completions to generate per input.
            use_steering:
                if True, apply the current steering plan (if any).
            steering_plan:
                optional SteeringPlan to use for this call only (overrides internal plan).
                If None, uses the internal plan.
            **gen_kwargs:
                additional kwargs passed to 'model.generate()'.

        returns:
            list of generated strings (length = len(inputs) * num_return_sequences).

        generation flow:
            notation:
                - Let B be batch size, T_in the (padded) input length, H the hidden size.
                - Decoder has L layers; we index user-facing layers as strings "1".. "L" (layer 1 is the first decoder block).
                - Steering plan maps layer names to one or more steering vectors with scales:
                '{"6": [SteeringVector(v6, scale=0.7)], "12": [SteeringVector(v12a, 1.0), SteeringVector(v12b, 0.4)]}'

        preparation:
            Given chat messages:
                msgs = [
                    {"role":"system","content":"Be concise."},
                    {"role":"user","content":"Two bullet points about koalas."}
                ]

            Encoding produces:
                - If chat template is available, 'apply_chat_template(..., tokenize=True)' yields `input_ids` of shape '[T1]'.
                - After 'tokenizer.pad([...])', the batch tensors have shapes:
                    - 'input_ids:  [B, T_in]'
                    - 'attention_mask: [B, T_in]'
          where 'T_in = T1' and 'B = 2' in this example.

        without steering:
            >>> wm = WisentModel("meta-llama/Meta-Llama-3-8B-Instruct", layers={}, device="cuda")
            >>> out_plain = wm.generate([msgs], max_new_tokens=32, use_steering=False)
            # out_plain: list[str] length B (or B * num_return_sequences)

            >>> for i, msg in enumerate(msgs):
            ...     print(f"User {i+1}: {msg['content']}")
            ...     print(f"Assistant {i+1}: {out_plain[i]}")

            internally during generation step 't = 0..T_out-1':
                - Each decoder block 'i' outputs a residual stream tensor of shape '[B, T_in + t, H]'.
                - No modification is applied; the model returns logits → token → appended to sequence.

        with steering (add AFTER layer i):
            # Build steering vectors of shape [H] for chosen layers; scales are per-vector.
            >>> plan = SteeringPlan.from_raw({
            ...     "6":  torch.randn(wm.hidden_size),   # will be normalized/broadcast if needed
            ...     "12": torch.randn(wm.hidden_size),
            ... }, scale=0.7, normalize=True)

           # Set once and use
            >>> wm.set_steering_from_raw({"6": plan.layers["6"][0].vector, "12": plan.layers["12"][0].vector},
                                 scale=0.7, normalize=True)


            What the hook 'sees' at a steered layer 'i' on step 't':
                - The layer's output (residual stream) 'h_i' has shape '[B, T_in + t, H]'.
                - Your steering vector 'v_i' is materialized to '[1, 1, H]' (or '[B,T,H]' if you passed that) and cast to the same dtype/device.
                - The hook returns 'h_i' = h_i + α_i * v_i' (if multiple vectors are configured for the same layer, it sums them).
                - This addition is cheap: one broadcasted add per steered layer, per step.

        
        shapes recap at generation step t (same for chat or plain strings):
        - Decoder block output:                '[B, T_in + t, H]'
        - Materialized steering vector:        '[1, 1, H]' (broadcast to '[B, T_in + t, H]')
        - Residual after steering (per layer): '[B, T_in + t, H]'

        example (one batch):
            >>> msgs = [
            ...   {"role":"system","content":"Be concise."},
            ...   {"role":"user","content":"Two bullet points about koalas."}
            ... ]
            >>> wm.apply_steering()  # or pass use_steering=True below
            >>> out = wm.generate([msgs], max_new_tokens=32, use_steering=True)
            >>> for i, msg in enumerate(msgs):
            ...     print(f"User {i+1}: {msg['content']}")
            ...     print(f"Assistant {i+1}: {out[i]}")
        """
        if use_steering:
            self.apply_steering(steering_plan)

        batch = self._batch_encode(inputs, add_generation_prompt=True)

        gen_out = self.hf_model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
            output_scores=False,
            **gen_kwargs,
        )

        if use_steering:
            self.detach()

        seqs = gen_out.sequences  # [B * num_return_sequences, T_total]
        texts = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)
        return texts

    @torch.inference_mode()
    def generate_with_stats(
        self,
        inputs: list[list[ChatMessage]],
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        collect_topk: int = 5,
        use_steering: bool = True,
        steering_plan: SteeringPlan | None = None,
        **gen_kwargs: Any,
    ) -> tuple[list[str], list[GenerationStats]]:
        """
        Generate with efficient per-token stats (logits / probs), compatible with steering.
        Implementation detail: uses `output_scores=True` + `return_dict_in_generate=True` (HF standard).  :contentReference[oaicite:11]{index=11}

        attributes:
            inputs:
                list of chat messages (each a list of {'role','content'} dicts).
            max_new_tokens:
                max tokens to generate (beyond the prompt).
            temperature:
                sampling temperature (0 = greedy, 1 = default sampling).
            top_p:
                nucleus sampling probability (0 = no filtering, 1 = full filtering).    
            do_sample:
                if False, uses greedy decoding (top_k=1).
            num_return_sequences:
                number of completions to generate per input.
            collect_topk:
                if > 0, collect top-k logits/probs per step for analysis/visualization.
            use_steering:
                if True, apply the current steering plan (if any).
            steering_plan:
                optional SteeringPlan to use for this call only (overrides internal plan).
                If None, uses the internal plan.
            **gen_kwargs:
                additional kwargs passed to 'model.generate()'.

        returns:
            - list of generated strings (length = len(inputs) * num_return_sequences).
            - list of GenerationStats (length = len(inputs) * num_return_sequences).
              Each GenerationStats has:
                tokens:
                    list of generated token ids (length = actual generated tokens).
                per_step:
                     if collect_topk > 0, list of TopLogits (length = actual generated tokens).
                    Each TopLogits has:
                        token_id:
                            the generated token id at that step.
                        logit:
                            the raw logit for that token.
                        prob:
                            the softmax probability for that token.
                        topk_ids:
                            if collect_topk > 0, list of top-k token ids at that step.
                        topk_probs:
                            if collect_topk > 0, list of top-k probabilities at that step.

        example:
            >>> msgs = [[
            ...   {"role":"system","content":"Be concise."},
            ...   {"role":"user","content":"Two bullet points about koalas."}
            ... ]]
            >>> wm = WisentModel("meta-llama/Meta-Llama-3-8B-Instruct", layers={}, device="cuda")
            >>> wm.set_steering_from_raw({"6": torch.randn(wm.hidden_size), "12": torch.randn(wm.hidden_size)}, scale=0.7, normalize=True)
            >>> texts, stats = wm.generate_with_stats(
            ...   msgs,
            ...   max_new_tokens=48, collect_topk=5, use_steering=True
            ... )
            >>> stats[0].per_step[0].prob  # probability of the first generated token
        """
        if use_steering:
            self.apply_steering(steering_plan)

        batch = self._batch_encode(inputs, add_generation_prompt=True)

        out = self.hf_model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
            output_scores=True,
            **gen_kwargs,
        )

        if use_steering:
            self.detach()

        texts = self.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)

        scores: list[torch.Tensor] = list(out.scores or [])
        stats: list[GenerationStats] = []

        if scores:
            stacked = torch.stack(scores, dim=0)             # [steps, B*num_ret, V]
            steps = stacked.size(0)
            gen_token_ids = out.sequences[:, -steps:]        # [B*num_ret, steps]

            logprobs = torch.log_softmax(stacked.float(), dim=-1)  # [steps, B, V]
            B = logprobs.size(1)
            V = logprobs.size(2)

            for b in range(B):
                toks = gen_token_ids[b].tolist()
                per_step: list[TopLogits] = []
                for t, tok_id in enumerate(toks):
                    lp_row = logprobs[t, b]                        # [V]
                    logit = scores[t][b, tok_id].item()
                    prob = float(lp_row[tok_id].exp().item())
                    if collect_topk > 0:
                        topk_vals, topk_ids = lp_row.topk(min(collect_topk, V))
                        per_step.append(TopLogits(
                            token_id=int(tok_id),
                            logit=float(logit),
                            prob=float(prob),
                            topk_ids=topk_ids.tolist(),
                            topk_probs=topk_vals.exp().tolist(),
                        ))
                    else:
                        per_step.append(TopLogits(
                            token_id=int(tok_id),
                            logit=float(logit),
                            prob=float(prob),
                        ))
                stats.append(GenerationStats(tokens=toks, per_step=per_step))
        else:
            for _ in range(out.sequences.size(0)):
                stats.append(GenerationStats(tokens=[], per_step=None))

        return texts, stats


    def set_steering_from_raw(self, raw: RawActivationMap, *, scale: float = 1.0, normalize: bool = False) -> None:
        """
        Replace the internal steering plan using a RawActivationMap (layer_name -> tensor).

        arguments:
            raw:
                RawActivationMap of steering vectors (layer_name -> tensor).
            scale:
                global scale applied to all vectors (default 1.0).
            normalize:
                if True, each vector is normalized to unit norm before use (default False).
        
        example:
            >>> wm.set_steering_from_raw({"6": torch.randn(wm.hidden_size)}, scale=0.5, normalize=True)
        """
        if not raw:
            self._steering_plan = SteeringPlan({})
            return

        report = run_control_vector_diagnostics(raw)
        for issue in report.issues:
            log_method = logger.error if issue.severity == "critical" else logger.warning
            log_method(
                "[control_vector diagnostics] %s (details=%s)",
                issue.message,
                issue.details,
            )

        if report.has_critical_issues:
            raise ValueError("Control vector diagnostics found critical issues; refusing to set steering.")

        self._steering_plan = SteeringPlan.from_raw(raw, scale=scale, normalize=normalize)

    def clear_steering(self) -> None:
        """
        Remove any existing steering configuration and active hooks.
        After calling this, generation is vanilla.
        """
        self._steering_plan = SteeringPlan({})
        self.detach()