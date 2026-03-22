
# Implementation notes and Log

Going from recent to first
---

### Depth Recurrence (EXPERIMENTAL)
  - 6 unique blocks run 2 times = 12 effective layers, half the stored params
  - Learnable pass_gate [2, 512] scales input differently per pass so model can tell them apart
  - U-Net skip connections reset each pass (collected and consumed within each loop)
  - Tunable: NUM_LAYERS=8 RECURRENCE=2 (16 eff.) or NUM_LAYERS=4 RECURRENCE=3 (12 eff.)
  - Risk: shared weights less expressive than unique — no leaderboard submission has tried this
  - All top submissions went the opposite direction: more unique layers via better compression
  - Need to A/B test against NUM_LAYERS=11 RECURRENCE=1 to see if it actually helps

### SmearGate
  - One parameter per dimension (512 floats), initialized to zero (sigmoid(0) = 0.5 = equal mix)
  - Applied after RMS norm, before transformer blocks
  - `x = (1 - gate) * current_token + gate * previous_token`
  - Gives the model cheap 1-token context before attention kicks in
  - Gate param added to scalar optimizer, listed in control tensor patterns so it stays fp32

### BigramHash                                                                                                
  - 10240 buckets, 128-dim embeddings projected to 512 (model_dim)              
  - XOR hash of consecutive token pairs
  - Learned scale starting at 0.05                                              
  - Weights initialized to zero so it gradually learns                          
  - Embedding goes to token optimizer (Adam), projection to Muon with the other matrix params                                                                 
  - Quantized as int8 in the export  

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