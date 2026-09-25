# LoRA Model - Java & DSA Tutor

This project is a small learning implementation of LoRA (Low-Rank Adaptation). It specializes a base language model into a beginner-friendly Java and DSA tutor.

Architecture:
Dataset -> Base LLM -> Tokenizer -> LoRA/PEFT -> Train adapter -> LoRA adapter
User prompt -> Base LLM + LoRA adapter -> Specialized tutor

LoRA freezes the original model weights and learns small low-rank matrices for selected layers. Conceptually: W' = W + BA.

Project structure:
data/cs_tutor.jsonl - starter training examples
training/train.py - LoRA training
training/config.yaml - configuration
inference/local_inference.py - local base + adapter inference
inference/openrouter.py - OpenRouter API inference
evaluation/evaluate.py - evaluation helper
docs/architecture.md - architecture explanation

Setup:
python -m venv .venv
Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

Train:
python training/train.py --config training/config.yaml

Run local inference:
python inference/local_inference.py --adapter outputs/lora-adapter

OpenRouter is used only as a hosted inference/API layer here; it does not perform local LoRA training. Put the API key in .env and never commit it.

Do not treat training loss as an accuracy percentage. Compare fixed prompts before and after LoRA for correctness, Java API accuracy, DSA reasoning, tutoring style, and hallucinations.