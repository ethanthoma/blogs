---
title: "Grad accum in Tinygrad"
date: "2025-09-26T23:10:00-07:00"
slug: "grad_accum_tinygrad"
---

# Grad Accumulation in Tinygrad

> tldr; realize your tensors or memory go OOM

Gradient accumulation let's you perform larger batch sizes during training when
memory constrained. The idea is simple: you run micro-batches, accumulate the
gradients, and then perform your backwards pass.

In PyTorch, people typically spit out something like so:

```python
accum: int
for i, (x, y) in enumerate(loader, 1):
    yhat = model(x)
    loss = loss_fn(yhat, y) / accum
    loss.backward()
    if i % accum == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

This impl is, often, wrong for cross-entropy with padding.

I say often because unless your micro-batch size is 1 or every sample is equal
token length, then this average (divide by `accum`) of averages (`loss_fn`
_probably_ returns average loss of micro-batch) is almost never equal to the
actual average.

To fix this, you need to (a) count the actual tokens, (b) change the reduction
method for your `loss_fn`, and (c) do the average manually:

```python
accum: int
pad_token: int
count: int = 0
loss: Tensor = 0
for i, (x, y) in enumerate(loader, 1):
    yhat = model(x)
    loss += loss_fn(yhat, y, reduction="sum")
    count += sum(y != pad_token)
    if i % accum == 0:
        (loss / count).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

In Tinygrad, if you do something similar it'll OOM. Which sucks since grad
accumulation is literally meant to fix that issue.

The problem is that Tinygrad is lazy. This means, each loop your code builds up
a compute graph for all the forward passes. Only when you actually realize the
model, after accumulating every grad step, does it try to compute the grads
effectively making the grad accum logic do nothing.

To solve this, all you have to do is just realize the tensors:

```python
accum: int
pad_token: int
count: Tensor = Tensor(0)
loss: Tensor = Tensor(0)
for i, (x, y) in enumerate(loader, 1):
    yhat = model(x)
    loss = loss.add(loss_fn(yhat, y, reduction="sum")).realize()
    count = count.add((y != pad_token).sum()).realize()
    if i % accum == 0:
        loss.div(count).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss, count = Tensor(0), Tensor(0)
```

Easy!
