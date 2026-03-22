
# Implementation notes and Log

Going from recent to first
---

### 

### Quicker wins

Bump to 10-11 layers       
3x MLP expansion. The SOTA uses 3x, lets follow suite     
Weight decay on Muon (0.04) as the baseline has none                
Gradient clipping (0.3) to stabalise training                
Longer warmdown (3000 vs 1200) hopefully for convergence                
Higher Muon momentum (0.99 vs 0.95) copy from SOTA                     
Magnitude pruning before quantization, compression ratio

### Int4 instead of int5
This aim is to reduce the model size so we can slam more things in there. needs to be QAT

QAT hyperparameters (lines ~89-90) — QAT_ENABLED=1 and QAT_CLIP_RANGE=7  (int4)                                                          
                                                                                
CastedLinear now has STE fake-quantization — during training, MLP weights see int4 quantization noise but gradients flow through unmodified           
                                                                                
Mixed post-training quantization — replaces uniform int8 with:             
  - Int4 (clip_range=7) for MLP weights — half the bits of int8                 
  - Int8 for attention weights — precision-sensitive                            
  - FP16 for embeddings — small, needs full precision                           
                                                                                
QAT activation — after model creation, all block.mlp.fc and block.mlp.proj layers get QAT enabled          