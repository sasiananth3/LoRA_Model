import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--prompt',
        default='Explain Java HashMap in simple terms.'
    )
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype
    ).to(device)

    text = '### Instruction:\\n' + args.prompt + '\\n### Response:\\n'
    inputs = tokenizer(text, return_tensors='pt').to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == '__main__':
    main()
