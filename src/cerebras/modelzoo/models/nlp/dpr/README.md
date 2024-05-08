# Dense Passage Retrieval (DPR)

This directory contains the Pytorch ML reference for the DPR model.

## Overview of the model

### DPR 

Dense Passage Retriever (DPR) is a technique introduced by Facebook Research that marked one of the biggest early successes in applying neural networks for retrieval. The goal of retrieval is to find passages that will help in answering a question. To accomplish this, DPR is composed of two sub-models: a question encoder, and a passage encoder. The idea is that questions and passages have different properties, so we can optimize each 
model (usually a BERT-based model) for each sub-domain. 

<p align="center">
<img src="./images/dpr_diagram.png" style="width: 75%; height: auto;">
</p>

During training, DPR incentives the two encoders to create embeddings of questions and passages, where useful passages are *close* to a question in the embedding-space, and less useful passages are *farther* away from the question. The model uses *contrastive loss* to maximize the similarity of a question with its corresponding passage, and minimizes the similarity of questions with non-matching passages. 

We currently support DPR **training** on CS-X, and inference can be run on GPUs. 
After training, you can create embeddings for all the passages in your data using the passage
encoder, and combine these into a vector database such as [FAISS](https://github.com/facebookresearch/faiss). The retrieval inference process is then 
to take a new question, run inference using the question encoder to get a question embedding, and retrieve the most similar passage embeddings from the vector database.  


### Contrastive loss

**Contrastive loss**: Contrastive loss has existed for decades, but gained popularity with the landmark paper by OpenAI paper on Contrastive Language-Image Pretraining, or [CLIP](https://arxiv.org/abs/2103.00020). DPR uses the same technique as CLIP, but between questions and passages instead of images and captions. Since the introduction of DPR, models trained with contrastive loss have become standard for retrieval. The current state-of-the-art retrievers have remained remarkably similar to the original recipe outlined by DPR. 

**Hard negatives**: Recall that the contrastive loss paradigm tries to do two things simultaneously: (1) maximize similarity between matching question-passage pairs (positive pairs), and (2) minimize similarity between non-matching question-passage pairs (negative pairs). Neural networks use batches of data for computational efficiency, so it is common practice to exploit this in creating non-matching pairs, by comparing a question with the passage of a *different* question from the same batch (in-batch negatives).  

Some datasets will additionally add *hard-negatives* for each question. Creating negatives within a batch is easy and efficient; however to obtain best performance, it is best to find passages that are *similar* to the positive passage, but do not contain the information reqired. These passages are called hard-negatives, as it is much more difficult to discern between this negative and the true positive passage. 


## DPR features dictionary 

To support the contrastive loss modeling paradigm, the features dictionary for DPR is different than most of the other models that we support. It is similar to the features dictionary for the BERT model, as the sub-encoders in DPR are based on BERT. 

- `questions_input_ids`: Input token IDs, padded with `0` to `max_sequence_length`.
  - Shape: `(batch_size, max_sequence_length)`
  - Type: `torch.int32`
- `questions_attention_mask`: Mask for padded positions. Has values `0` on the padded positions and `1` elsewhere.
  - Shape: `(batch_size, max_sequence_length)`
  - Type: `torch.int32`
- `questions_token_type_ids`: Segment IDs. Has values `0` on the positions corresponding to the first segment, and `1` on positions corresponding to the second segment. This is included because DPR uses BERT-based encoders, but there is usually only one segment in DPR.
  - Shape: `(batch_size, max_sequence_length)`
  - Type: `torch.int32`   
- `ctx_input_ids`: Input token IDs, padded with `0` to `max_sequence_length`. The second dimension corresponds to the positive and hard-negative per example.
  - Shape: `(batch_size, 2, max_sequence_length)`
  - Type: `torch.int32`
- `ctx_attention_mask`: Mask for padded positions. Has values `0` on the padded positions and `1` elsewhere. The second dimension corresponds to the positive and hard-negative per example.
  - Shape: `(batch_size, 2, max_sequence_length)`
  - Type: `torch.int32`
- `ctx_token_type_ids`: Segment IDs. Has values `0` on the positions corresponding to the first segment, and `1` on positions corresponding to the second segment. This is included because DPR uses BERT-based encoders, but there is usually only one segment in DPR. The second dimension corresponds to the positive and hard-negative per example. 
  - Shape: `(batch_size, 2, max_sequence_length)`
  - Type: `torch.int32`   


## Configurations included for this model

In order to train the model, you need to provide a yaml config file. We provide a sample yaml [configs](configs/) file for a popular configuration as reference. 

- [params_dpr_base_nq.yaml](./configs/params_dpr_base_nq.yaml) 

You can also adapt configs and create new ones based on your needs, particularly in specifying new datasets. 

## References

**Reference**: Karphukhin, Vladimir, et al. (2020). [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)

**Reference**: Radford, Alec, et al. (2021). [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020.pdf)

**Reference**: Douze, Matthijs, et al. (2024). [The Faiss Library](https://arxiv.org/abs/2401.08281)

**Reference**: Figure adapted from PyTorch forum [post](https://discuss.pytorch.org/t/loss-not-decreasing-for-a-network-for-finding-related-text/114233)
