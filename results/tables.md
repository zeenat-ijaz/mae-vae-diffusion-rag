## Table 1 - Data
| Item | Value |
|---|---|
| EuroSAT train / test images; labelled images used | 16200 / 5400; 160 |
| UHaT classes; train / test images | 40; 28328 / 4747 |
| BioASQ abstracts; chunks; test questions | 40181; 79332; 707 |
| Mean relevant abstracts per question | 8.66 |

## Table 2 - Training
| Model | Parameters | Final loss | Time / GPU |
|---|---|---|---|
| MAE | 3173248 | 0.0064 | 7.5 min / Tesla T4 |
| VAE | 220385 | 130.43 | 0.5 min / Tesla T4 |
| DDPM U-Net | 4028353 | 0.0229 | 34.1 min / Tesla T4 |

## Table 3 - Image results
| Item | Value |
|---|---|
| MAE classifier test accuracy | 0.5281 |
| VAE test loss = recon + KL | 130.62 = 104.54 + 26.09 |
| Diffusion blind test correct (member 1 / member 2, of 40) | 26 / 1 |
| Diffusion images both agreed on (of 40) | 3 |

## Table 4 - Retrieval
| hit@1 | hit@5 | hit@10 | recall@1 | recall@5 | recall@10 | mrr@10 |
|---|---|---|---|---|---|---|
| 0.6223 | 0.8161 | 0.8614 | 0.6223 | 0.5863 | 0.6024 | 0.7061 |

## Table 5 - Generation
| Item | Value |
|---|---|
| ROUGE-L | 0.2538 |
| Citation precision | 0.6774 |
| I don't know rate | 0.08 |
| Supported (member 1 / member 2, of 20) | 15 / 15 |
| Correct (member 1 / member 2, of 20) | 13 / 10 |
