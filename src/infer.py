from ops.time_util import time_in_ms, store_time
from src.utils import safe_print, read_stdin


def generate(transformer, tokenizer, sampler, prompt: str, steps: int):
    prompt = prompt or ""
    # encode prompt
    prompt_tokens = tokenizer.encode(prompt, bos=False, eos=False)
    if len(prompt_tokens) < 1:
        raise RuntimeError("Expected at least 1 prompt token")

    start_time = None

    token_list = prompt_tokens
    if len(prompt_tokens) > 1:
        transformer.prefill(prompt_tokens[:-1])

    token = prompt_tokens[-1]
    pos = len(prompt_tokens) - 1
    while pos < steps:
        logits = transformer.forward(token, pos)  # logits tensor [vocab_size]

        next_token = sampler.sample(logits)
        tz = time_in_ms()

        # print output
        token_list.append(next_token)
        token = next_token

        pos += 1

        # start timer after first iteration
        if start_time is None:
            start_time = time_in_ms()

        tp = time_in_ms()
        store_time('print_output', tp - tz)
        # stop if BOS token encountered
        if next_token == 1:
            break

    p = tokenizer.decode(token_list)
    safe_print(p)
    print()
    # flush to stdout immediately
    import sys
    sys.stdout.flush()

    # report tokens/sec
    if pos > 1:
        end_time = time_in_ms()
        tok_per_sec = (pos - 1) / ((end_time - start_time) / 1000.0)
        print(f"achieved tok/s: {tok_per_sec:.2f}")


# --------------------- Chat Loop ---------------------
def chat(transformer, tokenizer, sampler, cli_user_prompt=None, cli_system_prompt=None, steps=256):
    system_prompt = ""
    user_prompt = ""
    rendered_prompt = ""
    prompt_tokens = []
    user_turn = True
    pos = 0
    next_token = None
    user_idx = 0

    while pos < steps:
        if user_turn:
            if pos == 0:
                if cli_system_prompt is not None:
                    system_prompt = cli_system_prompt
                else:
                    system_prompt = read_stdin("Enter system prompt (optional): ")

            if pos == 0 and cli_user_prompt is not None:
                user_prompt = cli_user_prompt
            else:
                user_prompt = read_stdin("User: ")

            if pos == 0 and system_prompt.strip():
                rendered_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
            else:
                rendered_prompt = f"[INST] {user_prompt} [/INST]"

            prompt_tokens = tokenizer.encode(rendered_prompt, bos=True, eos=False)
            user_idx = 0
            user_turn = False
            print("Assistant: ", end="", flush=True)

        # select token to feed into transformer
        if user_idx < len(prompt_tokens):
            token = prompt_tokens[user_idx]
            user_idx += 1
        else:
            token = next_token

        # EOS token ends Assistant turn
        if token == 2:
            user_turn = True

        logits = transformer.forward(token, pos)
        next_token = sampler.sample(logits)
        pos += 1

        # print output
        if user_idx >= len(prompt_tokens) and next_token != 2:
            piece = tokenizer.decode(token, next_token)
            safe_print(piece)
        if next_token == 2:
            print()
