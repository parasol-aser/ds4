#!/usr/bin/env python3
"""Build ChatML prompts of about 1.4k, 4.3k, 8.6k, 17k and 33k tokens by
repeating the body of prompt_1439.txt inside one user turn."""
import os
out = os.environ.get('OUT', 'out'); os.makedirs(out, exist_ok=True)
t = open('prompt_1439.txt').read()
a = t.find('<|im_start|>user\n') + len('<|im_start|>user\n'); b = t.find('<|im_end|>', a)
body = t[a:b].strip(); tail = t[b:]
for name, reps in [('1k', 1), ('4k', 3), ('8k', 6), ('16k', 12), ('32k', 23)]:
    txt = '<|im_start|>user\n' + '\n\n'.join([body] * reps) + '\n\nSummarize the text above in one sentence.' + tail
    open(f'{out}/prompt_{name}.txt', 'w').write(txt)
print('prompts written to', out)
