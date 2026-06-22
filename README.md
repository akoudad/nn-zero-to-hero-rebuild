# nn-zero-to-hero-rebuild

im implementing core deep learning systems form scratch,starting with autograd and working ip to a GPT-2 reproduction.

## what we building ? 

- [x] **scalar autograd engine** - backprop from first principles, no frameworks
- [ ] **bigram + MLP language models**
- [ ] **manual backprop**
- [ ] **WaveNet**
- [ ] **GPT from scratch**
- [ ] **tokenizer**
- [ ] **GPT-2 reproduction**

## Structure 
\'\'\'
notebooks/ implementation per stage (Value class , manual backprop by hand first , then automated via backward(), topo sort , tanh activation ,graph , gradient checks)
engine.py same value class , extracted into a clean standalone module
\'\'\'

