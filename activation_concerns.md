# Activation Token Selection Comparison

## CAA Approach
- Captures activations from the **second-to-last token** (`-2`) of responses
- Implementation: `p_activations = p_activations[0, -2, :].detach().cpu()`

## wisent-guard Approach
- Always captures the **last token** (`-1`) of the sequence
- Implementation:
  ```python
  last_token_idx = hidden_states.shape[1] - 1
  self.layer_activations[layer_idx] = hidden_states[:, last_token_idx, :].detach().clone()
  ``` 