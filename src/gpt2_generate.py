"""GPT-2 generation with top-k / top-p sampling. transformers 3.x."""
import argparse

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def generate(prompt, model_name='gpt2-medium', max_length=120,
             top_k=50, top_p=0.92, temperature=0.9, n=3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()

    ids = tok.encode(prompt, return_tensors='pt').to(device)
    out = model.generate(
        ids,
        max_length=max_length,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        num_return_sequences=n,
        pad_token_id=tok.eos_token_id,
    )
    for i, seq in enumerate(out):
        print('--- sample %d ---' % (i + 1))
        print(tok.decode(seq, skip_special_tokens=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prompt', default='The most important security risk of LLMs is')
    p.add_argument('--model', default='gpt2-medium')
    args = p.parse_args()
    generate(args.prompt, model_name=args.model)
