# verbatim-llm

A Python library to understand and mitigate verbatim and near-verbatim memorization in Large Language Models (LLMs).

## Installation

```bash
pip install verbatim-llm
```

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from verbatim_llm import TokenSwapProcessor

# Load models
main_model_name = "EleutherAI/pythia-6.9b"
aux_model_name = "EleutherAI/pythia-70m"
device = "cuda"

aux_tokenizer = AutoTokenizer.from_pretrained(aux_model_name)
aux_model = AutoModelForCausalLM.from_pretrained(aux_model_name).to(device)
main_tokenizer = AutoTokenizer.from_pretrained(main_model_name)
main_model = AutoModelForCausalLM.from_pretrained(main_model_name).to(device)

# Initialize processor
processor = TokenSwapProcessor(aux_model, main_tokenizer)

# Generate with memorization mitigation
prompt = "The fox jumped over"
inputs = main_tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = main_model.generate(
    inputs.input_ids, 
    logits_processor=[processor], 
    max_new_tokens=50, 
    do_sample=False
)

result = main_tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## Features

- **TokenSwap**: Lightweight method to disrupt memorized sequences during generation
- More memorization mitigation methods coming soon

## Citation

If you use this library, please cite:

```bibtex
@article{prashant2025lightweight,
  title={A Lightweight Method to Disrupt Memorized Sequences in LLM},
  author={Prashant, Parjanya Prajakta and Ponkshe, Kaustubh and Salimi, Babak},
  journal={arXiv preprint arXiv:2502.05159},
  year={2025}
}
```

## License

MIT License
