# nn-zero-to-hero-rebuild

im implementing core deep learning systems from scratch,starting with autograd and working up to a GPT-2 reproduction.

## What is this

An **autograd engine** automatically computes gradients (derivatives) for you — instead of deriving calculus by hand for every operation in a neural net, we build a computation graph, and the engine walks it backward applying the chain rule to get every gradient. This is the core mechanism every deep learning framework (PyTorch, TensorFlow) is built on.

**micrograd** is Andrej Karpathy's minimal implementation of this idea — a scalar-valued autograd engine that proves the concept without any of PyTorch's complexity. This repo rebuilds it from scratch, then progresses through the rest toward a full GPT-2 reproduction.

## what we building ? 

- [x] **scalar autograd engine** - backprop from first principles, no frameworks
- [ ] **bigram + MLP language models**
- [ ] **manual backprop**
- [ ] **WaveNet**
- [ ] **GPT from scratch**
- [ ] **tokenizer**
- [ ] **GPT-2 reproduction**

## Structure
```
notebooks/    Value class — manual backprop by hand first, then automated
              via backward(), topo sort, tanh activation, graph, gradient checks
engine.py     same Value class, extracted into a clean standalone module
```

im putting my cat instead of Karpathy’s puppet lol
<img width="1179" height="1158" alt="99IMG_8614" src="https://github.com/user-attachments/assets/4a66c858-1a76-4240-b83e-c81df3c15d08" />
