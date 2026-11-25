# Analysis

All numbers come from results/tables.md and the figures in this folder.

## E.1 MAE: why mask 75% of an image?

Neighbouring patches of an image look very similar (grass next to grass, water next to water). If only a few patches are hidden, the model can copy the answer from the neighbours and learns almost nothing. With 75% masked (48 of 64 patches) the missing patches cannot be guessed from the next patch, so the encoder has to learn what the scene looks like as a whole (land use, texture, layout). Masking this much also makes training cheaper, because the encoder only sees the 16 visible patches. The pretraining loss fell from 0.0530 to 0.0064 in 100 epochs (7.5 min on a T4). A frozen encoder with 160 labelled images reached 0.5281 test accuracy, against 0.10 for chance, so the features are useful even though no labels were used for pretraining.

## E.2 MAE: which classes are confused most often?

The five most confused pairs on the 5400 test images (true class, predicted class):

| Count | True | Predicted |
|---|---|---|
| 153 | SeaLake | Pasture |
| 127 | PermanentCrop | HerbaceousVegetation |
| 114 | AnnualCrop | PermanentCrop |
| 107 | Highway | River |
| 97 | River | Highway |

Most pairs are classes that look alike at 64x64 pixels: the three vegetation classes share green and brown textures, and a highway and a river are both a thin line across the image. With only 160 labels, the linear classifier cannot separate these fine differences. The matrix is in fig_mae_confusion.png.

## E.3 VAE: reconstruction and KL during training, and what happens without the KL term

Over 30 epochs the training reconstruction loss fell from 190.0 to 105.0 and the KL term rose from 14.3 to 25.4 (fig_vae_loss.png). The model first learns to reconstruct, and as it stores more information in the latent code the KL grows and then flattens. The test loss ended at 130.62 (104.54 reconstruction + 26.09 KL), very close to the training loss, so there is no sign of overfitting.

We did not train a version without the KL term, so the following is the expected behaviour and not a measured result: without KL the loss is only reconstruction, so the encoder is free to place each image anywhere in the latent space and to make the variance tiny. It becomes a plain autoencoder. Reconstructions would look sharper, but the latent space would have gaps, and sampling z from N(0, I) would give poor images.

## E.4 VAE vs diffusion samples

The VAE samples (fig_vae_samples.png) are smooth, blurry strokes and many do not look like a clear letter. This is typical, because the pixel loss averages over many possible images. The diffusion samples (fig_diffusion_samples.png) have sharp, thin strokes and each row looks like its character, and in the blind test a member recognised 26 of 40 diffusion images. The price is speed and size: the VAE trained in 0.5 min with 220,385 parameters and samples in one step, while the diffusion model needed 34.1 min with 4,028,353 parameters and 1000 denoising steps for every sample. The diffusion model is also class conditional, which the VAE here is not.

## E.5 Diffusion: which characters are written worst?

In the blind test member 1 recognised 26 of 40 images and member 2 recognised 1 of 40, and the two members agreed on 3 images. The 14 images member 1 got wrong were: zwaad, seey, haa3, zwaa, khaa, taa, qaaf, kaaf, rhraa, haa2, zaal, choti yaa, bari yaa and meem. Many of these are letters that share a body shape and differ only in the dots or in a small mark (taa / ttaa, zwaa / zwaad, haa1 / haa2 / haa3, kaaf / qaaf). On the 28x28 images the dots are only one or two pixels, so the model draws the shape but the dots are often wrong or missing. The grid in fig_diffusion_samples.png shows the same: the rows for haa2, haa3 and hamza are the least clear. The big difference between the two members shows that the task depends strongly on how well the person knows the Urdu letters.

## E.6 RAG: three questions where retrieval failed

Of 707 questions, 98 had no relevant abstract in the top 10 (retrieval_failures.json holds examples).

1. "Is capmatinib effective for glioblastoma?" The relevant abstract starts with a conflict of interest statement, so its chunks carry little information about the question. The retriever returned a general paper about glioma treatment.
2. "What is F105-P?" This is a rare name that the small embedding model has never seen as a word. The top result was a paper about FOP, which has a similar sounding abbreviation. The relevant abstract about the protamine-antibody fusion protein was not found.
3. "Which main viral protein is targeted by the drug remdesivir?" The retriever returned a paper about liver transporters (drug handling), because the question is general and the match was on drug related words. The relevant paper is about Ebola virus polymerase inhibitors.

The common cause is that a small general embedding model matches on topic words and not on exact names or on the precise question.

## E.7 Why is Recall@k lower than Hit@k for the same k?

Hit@k is 1 when at least one relevant abstract is in the top k. Recall@k is the number of relevant abstracts found in the top k divided by min(k, number of relevant abstracts), so a question only gets full recall when all the places in the top k are relevant. A question has on average 8.66 relevant abstracts, so one correct abstract in the top k is enough for Hit@k, but it gives only 1/k for Recall@k. For k = 1 the two are equal (0.6223 and 0.6223), because the divisor is 1. At k = 5 we get Hit@5 0.8161 and Recall@5 0.5863, and at k = 10 we get Hit@10 0.8614 and Recall@10 0.6024. Recall@k can never be higher than Hit@k, since any question with recall above 0 also has a hit. Because the divisor grows with k, Recall@k can also go down when k grows: in fig_rag_recall.png it falls from 0.622 at k = 1 to 0.586 at k = 5 and 6, and then rises again to 0.638 at k = 20.

## Human evaluation (RAG answers, 20 items)

Member 1: 15 answers supported by the passages and 13 correct. Member 2: 15 supported and 10 correct. A partly correct answer was counted as not correct. The answers marked wrong were mostly the cases where the model said "I do not know" or where retrieval had returned unrelated passages (for example items 38, 45, 64 and 74).
