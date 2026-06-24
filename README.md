# nn-zero-to-hero-rebuild

im implementing core deep learning systems from scratch,starting with autograd and working up to a GPT-2 reproduction.

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
