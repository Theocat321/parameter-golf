
# Run Tracker

## Our Runs

| Run ID | Date | BPB | Size (MB) | Config | Notes |
|--------|------|-----|-----------|--------|-------|
| run-001 | 2026-03-22 | 1.3477 | 14.04 | 11L unique, int4 QAT, all quick wins | 1xH100, 1037 steps — undertrained |
| run-002 | 2026-03-22 | 1.4019 | 10.27 | 8L x2 recurrence, int4 QAT, all quick wins | 1xH100, 885 steps — WORSE, slower, kill this approach |
| run-003 | 2026-03-22 | 1.4781 | 13.67 | 11L + SWA, int4 QAT, all quick wins | 1xH100, 1038 steps — SWA hurt post-quant BPB (1.3479 pre-quant vs 1.4781 post-quant) |

---

## Run Log

### Run: run-001
**Date:** 2026-03-22
**GPU:** 1xH100
**Config:**
```
NUM_LAYERS=11
RECURRENCE=1
MLP_MULT=3.0
QAT_ENABLED=1
QAT_CLIP_RANGE=7
BIGRAM_VOCAB_SIZE=10240
BIGRAM_DIM=128
MUON_MOMENTUM=0.99
MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500
WEIGHT_DECAY=0.04
GRAD_CLIP_NORM=0.3
WARMDOWN_ITERS=3000
TRAIN_SEQ_LEN=1024
```
**Result:** BPB=1.3477  |  Size=14.04 MB (under 16MB cap, ~2MB headroom)
**Train time:** 600s (hit wallclock cap)
**Steps:** 1037/20000 @ 578ms/step
**Peak memory:** 13906 MiB
**Changes from baseline:** Int4 QAT on MLP, 11 layers (was 9), 3x MLP (was 2x), BigramHash, SmearGate, Muon WD 0.04, grad clip 0.3, warmdown 3000, momentum 0.99, magnitude pruning 3%
**Observations:**
  - Model severely undertrained — 1xH100 gets ~1037 steps vs ~20K expected on 8xH100
  - Warmdown never kicked in (needs 3000 iters, we barely passed 1000)
  - BPB 1.3477 worse than baseline 1.2244 but not a fair comparison — need 8xH100 or longer wallclock
  - Size is good: 14.04MB with 2MB to spare, int4 quantization working well (3.78x compression ratio)

### Run: run-002 (depth recurrence experiment)
**Date:** 2026-03-22
**Branch:** depth-recurrance-exp
**GPU:** 1xH100
**Config:**
```
NUM_LAYERS=8
RECURRENCE=2
MLP_MULT=3.0
QAT_ENABLED=1
QAT_CLIP_RANGE=7
BIGRAM_VOCAB_SIZE=10240
BIGRAM_DIM=128
MUON_MOMENTUM=0.99
WEIGHT_DECAY=0.04
GRAD_CLIP_NORM=0.3
WARMDOWN_ITERS=3000
TRAIN_SEQ_LEN=1024
```
**Result:** BPB=1.3976 (post-quant)  |  Size=10.27 MB
**Train time:** 600s (hit wallclock cap)
**Steps:** 885/20000 @ 678ms/step
**Peak memory:** 19705 MiB
**Observations:**
  - Worse than run-001 on every metric: higher BPB (1.3976 vs 1.3477), slower steps (678ms vs 579ms), fewer steps (885 vs 1037)
  - 19.7GB memory vs 13.9GB — recurrence doubles the compute graph through same weights
  - Only upside: 10.27MB size (lots of headroom) but that doesn't help if quality is worse
  - **Verdict: depth recurrence is a dead end. Unique layers are better.**

### Run: run-003 (SWA experiment)
**Date:** 2026-03-22
**Branch:** swa-experiment
**GPU:** 1xH100
**Config:** Same as run-001 + SWA_ENABLED=1, SWA_START_FRAC=0.4, SWA_EVERY=50
**Result:** BPB=1.4781 (post-quant), 1.3479 (pre-quant)  |  Size=13.67 MB
**Train time:** 600s (hit wallclock cap)
**Steps:** 1038/20000 @ 579ms/step
**Peak memory:** 13910 MiB
**SWA checkpoints averaged:** 20
**Observations:**
  - Pre-quant BPB (1.3479) nearly identical to run-001 (1.3477) — SWA didn't help or hurt training
  - Post-quant BPB (1.4781) much worse — averaged early undertrained checkpoints don't survive int4 quantization
  - SWA needs a fully trained model with proper warmdown to work. On 1xH100 with only 1038 steps it's harmful
  - **Verdict: keep SWA enabled for 8xH100 runs, but it's noise on 1xH100**

---

## A/B Tests To Run

| Test | Config A | Config B | Hypothesis |
|------|----------|----------|------------|
| Depth recurrence vs unique layers | NUM_LAYERS=6 RECURRENCE=2 | NUM_LAYERS=11 RECURRENCE=1 | Unique layers win but recurrence is cheaper |

## Best Config So Far
```
# paste winning env vars here once we have runs
```
