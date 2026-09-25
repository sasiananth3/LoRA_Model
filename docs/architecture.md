# LoRA Architecture

Training path:
1. Load a base causal language model.
2. Load Java and DSA instruction-response examples.
3. Tokenize the examples.
4. Freeze the base model.
5. Attach LoRA modules to selected attention projections.
6. Train the adapter parameters.
7. Save the adapter separately.

LoRA concept:
W' = W + BA

W is the original weight matrix. A and B are small trainable matrices. The base weights remain frozen.

Inference loads the base model and then attaches the saved adapter with PEFT.

OpenRouter is a separate hosted inference layer. It can be used to compare hosted models, but it is not the local LoRA training system.

For current or changing factual knowledge, retrieval/tools should be used instead of expecting LoRA training to keep information current.