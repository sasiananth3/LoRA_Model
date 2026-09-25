import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/cs_tutor.jsonl')
    args = parser.parse_args()
    with open(args.dataset, 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f'Evaluation prompts: {len(rows)}')
    print('Compare the same prompts on the base and LoRA models.')
    print('Check correctness, Java API accuracy, DSA reasoning, style, and hallucinations.')
    for i, row in enumerate(rows, 1):
        print(f'{i}. {row["instruction"]}')

if __name__ == '__main__':
    main()