import json
from pathlib import Path
from typing import TypeVar, Iterable, List, Union, Any
import random
import numpy as np
import torch
from tqdm.auto import tqdm
import os
import collections
import torch.nn.functional as F

NEGATIVE_INF = -100000.0

T = TypeVar('T')


def reduce_sum(value, mask, axis=None):
    if axis is None:
        return torch.sum(value * mask)
    return torch.sum(value * mask, axis)


def reduce_mean(value, mask, axis=None):
    if axis is None:
        return torch.sum(value * mask) / torch.sum(mask)
    return reduce_sum(value, mask, axis) / torch.sum(mask, axis)


def reduce_std(value, mask):
    return torch.sqrt(reduce_mean(torch.square(value), mask) - torch.square(reduce_mean(value, mask)))


def reduce_var(value, mask):
    return reduce_mean(torch.square(value), mask) - torch.square(reduce_mean(value, mask))


def logits_to_entropy(logits):
    distribution = torch.distributions.Categorical(logits=logits)
    return distribution.entropy()


def mask_pad(value, mask, pad_value=None):
    if pad_value is None:
        pad_value = NEGATIVE_INF
    return value * mask + pad_value * (1 - mask)


def clamp(value, min_value, max_value):
    return torch.max(torch.min(value, max_value), min_value)


def ceil_div(a, b):
    return (a - 1) // b + 1


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError('Inexact division: %s / %s = %s' % (a, b, a / b))
    return q


def whiten(values, masks, shift_mean=True, accelerator=None):
    if accelerator is not None:
        all_values = accelerator.gather(values) # (num_gpus * B, KL)
        all_masks = accelerator.gather(masks) # (num_gpus * B, KL)
        mean, var = reduce_mean(all_values, all_masks), reduce_std(all_values, all_masks)
    else:
        mean, var = reduce_mean(values, masks), reduce_std(values, masks)
    # if accelerator is not None and accelerator.is_main_process:
    #     print(f'all_values: {all_values}, all_masks: {all_masks}')
    #     print(f'mean: {mean}, var: {var}')
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def flatten_dict(nested, sep='.'):
    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, collections.Mapping):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v
    flat = {}
    rec(nested, '', flat)
    return flat


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    assert batch_size > 0

    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []

        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch


def set_seed(seed=19260817, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file) as f:
        for line in f:
            yield json.loads(line)


def load_cache(file: Path):
    if file.exists():
        with file.open() as f:
            for line in tqdm(f, desc=f'Loading cache from {file}'):
                yield json.loads(line)


def args_to_filename(args):
    return f'_reward-{args.reward_shape}'
    '''
    return "_klCoef" + str(args.kl_coef) + \
        "_lr" + str(args.lr) + \
        "_batchSize" + str(args.batch_size) + \
        "_eps" + str(args.total_episodes) + \
        "_temp" + str(args.temperature) + \
        "_initModel_" + str(args.init_model_type) + \
        "_refModel_" + str(args.ref_model_type) + \
        "_valModel_" + str(args.value_model_type) + \
        "_respLen" + str(args.response_length) + \
        "_realKL_" + str(args.real_kl)
    '''

def get_tensorboard_logname(comment=""):
    import socket
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(
        'runs', current_time + '_' + socket.gethostname() + comment)
    return log_dir


def append_string(input_ids, attention_mask, s, tokenizer):
    '''
    input_ids: (B, L)
    s: str
    returns (B, L+l-1), where l is length of s in tokens (including </s>)
    '''
    lens = attention_mask.sum(dim=1)
    token_ids = tokenizer.encode(s) # ends with </s>
    l = len(token_ids)
    new_input_ids = F.pad(input_ids, (0, l-1), value=tokenizer.pad_token_id)
    new_attention_mask = F.pad(attention_mask, (0, l-1), value=0)
    B = input_ids.size(0)
    for b in range(B):
        new_input_ids[b, lens[b]-1:lens[b]+l-1] = torch.tensor(token_ids, dtype=torch.long, device=new_input_ids.device)
        new_attention_mask[b, lens[b]-1:lens[b]+l-1] = 1
    return new_input_ids, new_attention_mask

def concat_strings(a_input_ids, a_attention_mask, b_input_ids, b_attention_mask, tokenizer):
    '''
    a_input_ids: (B, La)
    b_input_ids: (B, Lb)
    returns (B, La+Lb-1)
    '''
    a_lens = a_attention_mask.sum(dim=1)
    b_lens = b_attention_mask.sum(dim=1)
    lb = b_input_ids.size(1)
    new_input_ids = F.pad(a_input_ids, (0, lb-1), value=tokenizer.pad_token_id)
    new_attention_mask = F.pad(a_attention_mask, (0, lb-1), value=0)
    B = a_input_ids.size(0)
    for b in range(B):
        new_input_ids[b, a_lens[b]-1:a_lens[b]+b_lens[b]-1] = b_input_ids[b, :b_lens[b]]
        new_attention_mask[b, a_lens[b]-1:a_lens[b]+b_lens[b]-1] = 1
    return new_input_ids, new_attention_mask

def append_and_concat(a_input_ids, a_attention_mask, s, b_input_ids, b_attention_mask, tokenizer):
    '''
    a_input_ids: (B, La)
    b_input_ids: (B, Lb)
    s: str
    returns (B, La+l-1+Lb-1)
    '''
    a_lens = a_attention_mask.sum(dim=1)
    b_lens = b_attention_mask.sum(dim=1)
    lb = b_input_ids.size(1)
    token_ids = tokenizer.encode(s) # ends with </s>
    l = len(token_ids)
    new_input_ids = F.pad(a_input_ids, (0, l-1+lb-1), value=tokenizer.pad_token_id)
    new_attention_mask = F.pad(a_attention_mask, (0, l-1+lb-1), value=0)
    B = a_input_ids.size(0)
    for b in range(B):
        if b_lens[b] == 1:
            continue
        new_input_ids[b, a_lens[b]-1:a_lens[b]+l-1] = torch.tensor(token_ids, dtype=torch.long, device=new_input_ids.device)
        new_attention_mask[b, a_lens[b]-1:a_lens[b]+l-1] = 1
        new_input_ids[b, a_lens[b]+l-1-1:a_lens[b]+l-1+b_lens[b]-1] = b_input_ids[b, :b_lens[b]]
        new_attention_mask[b, a_lens[b]+l-1-1:a_lens[b]+l-1+b_lens[b]-1] = 1
    return new_input_ids, new_attention_mask
