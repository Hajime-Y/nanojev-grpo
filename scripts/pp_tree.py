"""Prefix-shared ("tree") forward for DecisionModel: one sequence per question instead of one per candidate.

The k candidate paths of a question share a token prefix (state + question). Causal attention lets the
prefix be computed once: each candidate suffix attends to the prefix and to itself only, and continues the
prefix positions. Logits equal DecisionModel.forward up to floating-point noise.
"""


def split_tree(leaves):
    shortest = min(map(len, leaves))
    shared = 0
    while shared < shortest - 1 and all(leaf[shared] == leaves[0][shared] for leaf in leaves):
        shared += 1
    return leaves[0][:shared], [leaf[shared:] for leaf in leaves]


def tree_tokens(example):
    prefix, suffixes = split_tree(example["leaf_tokens"])
    tokens, positions, blocks, ends = list(prefix), list(range(len(prefix))), [0] * len(prefix), []
    for index, suffix in enumerate(suffixes, 1):
        tokens += suffix
        positions += range(len(prefix), len(prefix) + len(suffix))
        blocks += [index] * len(suffix)
        ends.append(len(tokens) - 1)
    return tokens, positions, blocks, ends


def tree_forward(model, examples, pad_token):
    import torch
    import torch.nn.functional as F

    device = model.scalar.weight.device
    rows = [tree_tokens(ex) for ex in examples]
    width = max(len(row[0]) for row in rows)
    tokens = torch.full((len(rows), width), pad_token, dtype=torch.long, device=device)
    positions = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    blocks = torch.full((len(rows), width), -1, dtype=torch.long, device=device)
    for i, (ids, pos, blk, _) in enumerate(rows):
        tokens[i, :len(ids)] = torch.tensor(ids, device=device)
        positions[i, :len(ids)] = torch.tensor(pos, device=device)
        blocks[i, :len(ids)] = torch.tensor(blk, device=device)
    order = torch.arange(width, device=device)
    allowed = (order[None, :] <= order[:, None])[None] & (
        (blocks[:, None, :] == 0) | (blocks[:, None, :] == blocks[:, :, None]))
    hidden = model.backbone(input_ids=tokens, attention_mask=allowed[:, None], position_ids=positions,
                            use_cache=False).last_hidden_state
    kmax = max(len(ex["candidate_ids"]) for ex in examples)
    h = hidden.new_zeros((len(examples), kmax, hidden.shape[-1]))
    valid = torch.zeros((len(examples), kmax), dtype=torch.bool, device=device)
    for i, (ex, row) in enumerate(zip(examples, rows)):
        ends = torch.tensor(row[3], device=device)
        h[i, :len(ends)] = hidden[i, ends]
        valid[i, :len(ex["candidate_ids"])] = True
    # Head: identical to DecisionModel.forward after the backbone.
    h = model.norm(h)
    z = model.scalar(h).squeeze(-1).float()
    choice = torch.tensor([i for i, ex in enumerate(examples) if ex["type"] == "choice"], device=device)
    if model.set_head == "attention" and len(choice):
        log_k = valid[choice].sum(-1).float().log()[:, None, None].expand(-1, kmax, 1)
        u = model.set_project(torch.cat([h[choice], log_k.to(h.dtype)], dim=-1))
        mixed, _ = model.set_attention(u, u, u, key_padding_mask=~valid[choice], need_weights=False)
        delta = model.set_output(torch.tanh(u + mixed)).squeeze(-1).float()
        z = z.index_add(0, choice, delta)
    out = []
    for i, ex in enumerate(examples):
        if ex["type"] == "boolean":
            out.append(F.pad(torch.stack([z[i, 0] * 0, z[i, 0]]), (0, kmax - 2)))
        else:
            out.append(z[i])
    return torch.stack(out).masked_fill(~valid, -1e9), valid
