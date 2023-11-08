# Cerebras-GPT

This directory contains the configuration files necessary to reproduce the results in our [Cerebras-GPT Blog](https://www.cerebras.net/cerebras-gpt) and [arXiv paper](https://arxiv.org/abs/2304.03208) using CS-2 systems.

## Quickstart for Pre-Trained Checkpoints

We host the pre-trained checkpoints in a publicly accessible S3 bucket. You may continue training from these checkpoints as you would any saved Cerebras checkpoint. 

First, download the weights to your desired directory:

```bash
wget -O my_checkpoint_directory/my_checkpoint_name.mdl <URL>
```

Next, call `run.py` with the `--checkpoint_path` flag:

```bash
python run.py CSX {pipeline,weight_streaming} --checkpoint_path /path/to/checkpoint ... <remaining arguments>
```

| Model family | Parameters | Checkpoint URL |
| --- | --- | --- |
| Cerebras-GPT     | 111M       | [cerebras-gpt-dense-111m-sp-checkpoint_final.mdl](https://cerebras-public.s3.us-west-2.amazonaws.com/cerebras-gpt-checkpoints/dense/111M/sp/cerebras-gpt-dense-111m-sp-checkpoint_final.mdl)     |
| Cerebras-GPT     | 256M       | [cerebras-gpt-dense-256m-sp-checkpoint_final.mdl](https://cerebras-public.s3.us-west-2.amazonaws.com/cerebras-gpt-checkpoints/dense/256M/sp/cerebras-gpt-dense-256m-sp-checkpoint_final.mdl)     |
| Cerebras-GPT     | 590M       | [cerebras-gpt-dense-590m-sp-checkpoint_final.mdl](https://cerebras-public.s3.us-west-2.amazonaws.com/cerebras-gpt-checkpoints/dense/590M/sp/cerebras-gpt-dense-590m-sp-checkpoint_final.mdl)     |
| Cerebras-GPT     | 1.3B       | [cerebras-gpt-dense-1p3b-sp-checkpoint_final.mdl](https://cerebras-public.s3.us-west-2.amazonaws.com/cerebras-gpt-checkpoints/dense/1.3B/sp/cerebras-gpt-dense-1p3b-sp-checkpoint_final.mdl)     |
| Cerebras-GPT     | 2.7B       | [cerebras-gpt-dense-2p7b-sp-checkpoint_final.mdl](https://cerebras-public.s3.us-west-2.amazonaws.com/cerebras-gpt-checkpoints/dense/2.7B/sp/cerebras-gpt-dense-2p7b-sp-checkpoint_final.mdl)     |
| Cerebras-GPT     | 6.7B       | [cerebras-gpt-dense-6p7b-sp-checkpoint_final.mdl](https://cerebras-public.s3.us-west-2.amazonaws.com/cerebras-gpt-checkpoints/dense/6.7B/sp/cerebras-gpt-dense-6p7b-sp-checkpoint_final.mdl)     |
| Cerebras-GPT     | 13B        | [cerebras-gpt-dense-13b-sp-checkpoint_final.mdl](https://cerebras-public.s3.us-west-2.amazonaws.com/cerebras-gpt-checkpoints/dense/13B/sp/cerebras-gpt-dense-13b-sp-checkpoint_final.mdl)         |
| Cerebras-GPT muP | 111M muP   | [cerebras-gpt-dense-111m-mup-checkpoint_final.mdl](https://cerebras-public.s3.us-west-2.amazonaws.com/cerebras-gpt-checkpoints/dense/111M/mup/cerebras-gpt-dense-111m-mup-checkpoint_final.mdl) |
| Cerebras-GPT muP | 256M muP   | [cerebras-gpt-dense-256m-mup-checkpoint_final.mdl](https://cerebras-public.s3.us-west-2.amazonaws.com/cerebras-gpt-checkpoints/dense/256M/mup/cerebras-gpt-dense-256m-mup-checkpoint_final.mdl) |
| Cerebras-GPT muP | 590M muP   | [cerebras-gpt-dense-590m-mup-checkpoint_final.mdl](https://cerebras-public.s3.us-west-2.amazonaws.com/cerebras-gpt-checkpoints/dense/590M/mup/cerebras-gpt-dense-590m-mup-checkpoint_final.mdl) |
| Cerebras-GPT muP | 1.3B muP   | [cerebras-gpt-dense-1p3b-mup-checkpoint_final.mdl](https://cerebras-public.s3.us-west-2.amazonaws.com/cerebras-gpt-checkpoints/dense/1.3B/mup/cerebras-gpt-dense-1p3b-mup-checkpoint_final.mdl) |
| Cerebras-GPT muP | 2.7B muP   | [cerebras-gpt-dense-2p7b-mup-checkpoint_final.mdl](https://cerebras-public.s3.us-west-2.amazonaws.com/cerebras-gpt-checkpoints/dense/2.7B/mup/cerebras-gpt-dense-2p7b-mup-checkpoint_final.mdl) |

## Model Description

The Cerebras-GPT family is released to facilitate research into LLM scaling laws using open architectures and data sets and demonstrate the simplicity of and scalability of training LLMs on the Cerebras software and hardware stack. All Cerebras-GPT models are available on Hugging Face.

The family includes 111M, 256M, 590M, 1.3B, 2.7B, 6.7B, and 13B models.

All models in the Cerebras-GPT family have been trained in accordance with [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556) (20 tokens per model parameter) which is compute-optimal.

These models were trained on the [Andromeda](https://www.cerebras.net/andromeda/) AI supercomputer comprised of 16 CS-2 wafer scale systems. Cerebras' [weight streaming technology](https://www.cerebras.net/blog/linear-scaling-made-possible-with-weight-streaming) simplifies the training of LLMs by disaggregating compute from model storage. This allowed for efficient scaling of training across nodes using simple data parallelism.

Cerebras systems checkpoints for pre-training and fine tuning are available in the cloud via the [Cerebras Model Studio](https://www.cerebras.net/product-cloud/). Hugging Face compatible checkpoints are available on [our Hugging Face page](https://huggingface.co/models?sort=downloads&search=cerebras-gpt).

## Model Details
* Developed by: [Cerebras Systems](https://www.cerebras.net/)
* License: Apache 2.0
* Model type: Transformer-based Language Model
* Architecture: GPT-3 style architecture
* Data set: The Pile
* Tokenizer: Byte Pair Encoding
* Vocabulary Size: 50257
* Sequence Length: 2048
* Optimizer: AdamW, (β1, β2) = (0.9, 0.95), adam_eps = 1e−8 (1e−9 for larger models)
* Positional Encoding: Learned
* Language: English
* Learn more: [Cerebras-GPT Paper](https://arxiv.org/abs/2304.03208) for training procedure, config files, and details on how to use.

**Contact**: To ask questions about Cerebras-GPT models, join the Cerebras Discord, and post them in **#scaling-laws-release.**

*NOTE:* The `muP` configs will be available in the later Model Zoo release.
<br><br>

| Model         | Parameters | Layers | d_model | Heads | d_head | d_ffn  | LR       | BS (seq) | BS (tokens)     |
|---------------|------------|--------|---------|-------|--------|--------|----------|----------|----------------|
| Cerebras-GPT  | 111M       | 10     | 768     | 12    | 64     | 3072   | 6.00E-04 | 120      | 246K           |
| Cerebras-GPT  | 256M       | 14     | 1088    | 17    | 64     | 4352   | 6.00E-04 | 264      | 541K           |
| Cerebras-GPT  | 590M       | 18     | 1536    | 12    | 128    | 6144   | 2.00E-04 | 264      | 541K           |
| Cerebras-GPT  | 1.3B       | 24     | 2048    | 16    | 128    | 8192   | 2.00E-04 | 528      | 1.08M          |
| Cerebras-GPT  | 2.7B       | 32     | 2560    | 20    | 128    | 10240  | 2.00E-04 | 528      | 1.08M          |
| Cerebras-GPT  | 6.7B       | 32     | 4096    | 32    | 128    | 16384  | 1.20E-04 | 1040     | 2.13M          |
| Cerebras-GPT  | 13B        | 40     | 5120    | 40    | 128    | 20480  | 1.20E-04 | 720/1080 | 1.47M/2.21M    |
| Cerebras-GPT-muP | 111M    | 10     | 768     | 12    | 64     | 3072   | 6.00E-03 | 120      | 246K           |
| Cerebras-GPT-muP | 256M    | 14     | 1088    | 17    | 64     | 4352   | 6.00E-03 | 264      | 541K           |
| Cerebras-GPT-muP | 590M    | 18     | 1536    | 12    | 128    | 6144   | 6.00E-03 | 264      | 541K           |
| Cerebras-GPT-muP | 1.3B    | 24     | 2048    | 16    | 128    | 8192   | 6.00E-03 | 528      | 1.08M          |
| Cerebras-GPT-muP | 2.7B    | 32     | 2560    | 20    | 128    | 10240  | 6.00E-03 | 528      | 1.08M          |

<br><br>


## Included Configurations

- `111m.yaml`: a configuration to train the 111M parameter model on a single CS-2.
- `256m.yaml`: a configuration to train the 256M parameter model on a single CS-2.
- `590m.yaml`: a configuration to train the 590M parameter model on a single CS-2.
- `1p3b.yaml`: a configuration to train the 1.3B parameter model on a cluster of 4 CS-2 systems.
- `2p7b.yaml`: a confituration to train the 2.7B parameter model on a cluster of 4 CS-2 systems.
- `6p7b.yaml`: a configuration to train the 6.7B parameter model on a cluster of 8 CS-2 systems.
- `13b_bs720.yaml` and `13b_bs1080.yaml`: configurations to train the 13B parameter model. Due to pratical considerations, we started training on 8 CS-2 systems with a global batch size of 720 and finished training on 12 CS-2 systems with a global batch size of 1080. In particular we trained for the first 57k steps on 8 CS-2 systems using parameters file `13b_bs720.yaml`, reset the global step in the resulting checkpoint file to 38k (i.e. 57k * 720 / 1080), and finished training using the parameters file `13b_bs1080.yaml`. This procedure resulted in training for a total of 257B tokens.

## Training data

Cerebras-GPT is trained using [the Pile](https://pile.eleuther.ai) dataset from [EleutherAI](https://www.eleuther.ai) which consists of data from 22 data sources. See the [Pile paper](https://arxiv.org/abs/2101.00027) for a more detailed breakdown of data sources and methodology.

Recent works find significant duplicate data present in the Pile. Eleuther’s Pythia applies a deduplication process to reduce replicated data, decreasing the total token count by 33%. Our models are trained on the Pile **without deduplication**, which presents an opportunity for further improvement with the deduplicated data set.

Our tokenized version of the Pile has 371B tokens. We used byte-pair encoding, a vocabulary size of 50257, and a maximum sequence length of 2048. We include more details about the training dataset preprocessing in Appendix A.1 of our paper.

<br><br>

## Training procedure

We use the GPT-3 style model architecture. All of our layers use full attention as opposed to the GPT-3 style sparse banded attention. The model shapes were selected to either follow aspect ratio 80 or are the same shape as GPT-3 models. Learning rate warmed up for 375M tokens (1500 steps for 111M and 256M models) and 10x cosine decayed. No dropout was used and weight decay was set to 0.1. All models are trained with MSL of 2048.

All models were trained to Chinchilla point: 20x more tokens than model parameters. Number of steps changed based on fixed batch size (2048) and sequence length (varied by model). See Training Table, below, for detail. 

<br>

Model Params | Sequence Length | Batch Size | Number of Steps | Tokens | Tokens per Parameter | Flops
------------ | -------------- | ---------- | --------------- | ------ | -------------------- | -----
111M         | 2048           | 120        | 9037            | 2.22E+09 | 20                  | 2.5E+18
256M         | 2048           | 264        | 9468            | 5.12E+09 | 20                  | 1.1E+19
590M         | 2048           | 264        | 21836           | 1.18E+10 | 20                  | 5.3E+19
1.3B         | 2048           | 528        | 24334           | 2.63E+10 | 20                  | 2.5E+20
2.7B         | 2048           | 528        | 49041           | 5.30E+10 | 20                  | 9.8E+20
6.7B         | 2048           | 1040       | 62522           | 1.33E+11 | 20                  | 5.9E+21
13B          | 2048           | 720        | 174335          | 2.57E+11 | 20                  | 2.1E+22

<br><br>

## Evaluations

We evaluate our models on the PILE validation set comprising 380M tokens. We also evaluate the public checkpoints of Pythia, Eleuther (2022); OPT, Zhang et al. (2022); GPT-NeoX 20B, Black et al. (2022); and GPT-J 6B, Wang & Komatsuzaki (2021). We trained models from smallest to largest and fit a power law as we went along. The power law was helpful for extrapolating the validation loss of the next largest model we trained and provided confidence about whether the training run was going well.

We performed upstream (pre-training) evaluations of text prediction cross-entropy using the Pile validation and test splits. We performed downstream evaluations of text generation accuracy on standardized tasks using the [Eleuther lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Results are compared against many publicly available large language models in Section 3 of the paper.

#### 0-shot Evaluation
| Model          | Count | Training FLOPs | PILE test xent | Hella- Swag | PIQA  | Wino- Grande | Lambada | ARC-e | ARC-c | OpenBookQA | Downstream Average |
| -------------- | ----- | -------------- | -------------- | ----------- | ----- | ------------ | ------- | ----- | ----- | ---------- | ------------------ |
| Cerebras       | 111M  | 2.6E+18        | 2.608          | 0.268       | 0.594 | 0.488        | 0.194   | 0.380 | 0.166 | 0.118      | 0.315              |
|                | 256M  | 1.3E+19        | 2.349          | 0.274       | 0.613 | 0.511        | 0.293   | 0.410 | 0.170 | 0.158      | 0.347              |
|                | 590M  | 6.1E+19        | 2.181          | 0.291       | 0.627 | 0.498        | 0.366   | 0.464 | 0.190 | 0.158      | 0.370              |
|                | 1.3B  | 2.8E+20        | 1.997          | 0.325       | 0.664 | 0.521        | 0.462   | 0.508 | 0.224 | 0.166      | 0.410              |
|                | 2.7B  | 1.1E+21        | 1.834          | 0.386       | 0.701 | 0.559        | 0.567   | 0.571 | 0.246 | 0.206      | 0.462              |
|                | 6.7B  | 6.3E+21        | 1.704          | 0.447       | 0.739 | 0.602        | 0.636   | 0.643 | 0.282 | 0.238      | 0.512              |
|                | 13B   | 2.3E+22        | 1.572          | 0.513       | 0.766 | 0.646        | 0.696   | 0.714 | 0.367 | 0.286      | 0.570              |
| Cerebras muP   | 111M  | 2.6E+18        | 2.588          | 0.268       | 0.598 | 0.519        | 0.204   | 0.390 | 0.176 | 0.124      | 0.325              |
|                | 256M  | 1.3E+19        | 2.359          | 0.274       | 0.617 | 0.505        | 0.287   | 0.427 | 0.194 | 0.156      | 0.351              |
|                | 590M  | 6.1E+19        | 2.155          | 0.295       | 0.644 | 0.517        | 0.362   | 0.470 | 0.194 | 0.172      | 0.379              |
|                | 1.3B  | 2.8E+20        | 1.984          | 0.334       | 0.682 | 0.512        | 0.471   | 0.515 | 0.223 | 0.196      | 0.419              |
|                | 2.7B  | 1.1E+21        | 1.846          | 0.388       | 0.697 | 0.557        | 0.558   | 0.569 | 0.241 | 0.218      | 0.461              |
| Pythia         | 70M   | 1.6E+20        | 2.504          | 0.270       | 0.590 | 0.491        | 0.259   | 0.413 | 0.185 | 0.132      | 0.334              |
|                | 160M  | 4.1E+20        | 2.186          | 0.293       | 0.627 | 0.519        | 0.389   | 0.452 | 0.181 | 0.160      | 0.375              |
|                | 410M  | 1.1E+21        | 1.971          | 0.333       | 0.668 | 0.530        | 0.505   | 0.504 | 0.213 | 0.178      | 0.419              |
|                | 1B    | 2.2E+21        | 1.845          | 0.376       | 0.705 | 0.545        | 0.566   | 0.559 | 0.243 | 0.196      | 0.456              |
|                | 1.4B  | 3.2E+21        | 1.793          | 0.398       | 0.711 | 0.565        | 0.604   | 0.576 | 0.256 | 0.204      | 0.474              |
|                | 2.8B  | 6.1E+21        | 1.720          | 0.451       | 0.737 | 0.612        | 0.654   | 0.629 | 0.288 | 0.220      | 0.513              |
|                | 6.9B  | 1.4E+22        | 1.626          | 0.482       | 0.746 | 0.611        | 0.679   | 0.669 | 0.323 | 0.270      | 0.540              |
|                | 12B   | 2.4E+22        | 1.582          | 0.505       | 0.761 | 0.645        | 0.705   | 0.700 | 0.336 | 0.284      | 0.562              |
| Pythia deduped | 70M   | 1.6E+20        | 2.549          | 0.273       | 0.607 | 0.526        | 0.257   | 0.404 | 0.175 | 0.136      | 0.340              |
|                | 160M  | 4.1E+20        | 2.204          | 0.294       | 0.632 | 0.509        | 0.370   | 0.451 | 0.204 | 0.172      | 0.376              |
|                | 410M  | 1.1E+21        | 1.989          | 0.341       | 0.668 | 0.534        | 0.514   | 0.519 | 0.206 | 0.180      | 0.423              |
|                | 1B    | 2.2E+21        | 1.858          | 0.387       | 0.712 | 0.546        | 0.585   | 0.568 | 0.241 | 0.212      | 0.464              |
|                | 1.4B  | 3.2E+21        | 1.889          | 0.403       | 0.729 | 0.561        | 0.610   | 0.582 | 0.265 | 0.198      | 0.478              |
|                | 2.8B  | 6.1E+21        | 1.724          | 0.466       | 0.743 | 0.612        | 0.672   | 0.662 | 0.299 | 0.232      | 0.526              |
|                | 6.9B  | 1.4E+22        | 1.644          | 0.488       | 0.756 | 0.636        | 0.695   | 0.667 | 0.320 | 0.252      | 0.545              |
|                | 12B   | 2.4E+22        | 1.601          | 0.516       | 0.761 | 0.639        | 0.712   | 0.697 | 0.341 | 0.280      | 0.564              |
| NeoX           | 20B   | 6.4E+22        | 1.519          | 0.535       | 0.779 | 0.661        | 0.720   | 0.723 | 0.380 | 0.290      | 0.584              |
| GPT-J          | 6B    | 1.7E+22        | 1.613          | 0.518       | 0.752 | 0.640        | 0.683   | 0.670 | 0.340 | 0.288      | 0.556              |
| OPT            | 125M  | 4.1E+20        | \-             | 0.292       | 0.630 | 0.503        | 0.379   | 0.435 | 0.189 | 0.166      | 0.371              |
|                | 350M  | 1.1E+21        | \-             | 0.320       | 0.644 | 0.523        | 0.452   | 0.440 | 0.207 | 0.176      | 0.395              |
|                | 1.3B  | 3.2E+21        | \-             | 0.415       | 0.717 | 0.595        | 0.579   | 0.570 | 0.234 | 0.234      | 0.478              |
|                | 2.7B  | 6.1E+21        | \-             | 0.458       | 0.738 | 0.610        | 0.637   | 0.609 | 0.268 | 0.250      | 0.510              |
|                | 6.7B  | 1.4E+22        | \-             | 0.505       | 0.763 | 0.654        | 0.677   | 0.656 | 0.307 | 0.276      | 0.548              |
|                | 13B   | 2.7E+22        | \-             | 0.524       | 0.759 | 0.651        | 0.687   | 0.671 | 0.329 | 0.270      | 0.556              |



#### 5-shot Evaluation
| Model          | Count | Hella- Swag | PIQA  | Wino- Grande | Lambada | ARC-e | ARC-c | OpenBookQA | Downstream Average |
| -------------- | ----- | ----------- | ----- | ------------ | ------- | ----- | ----- | ---------- | ------------------ |
| Cerebras       | 111M  | 0.267       | 0.588 | 0.475        | 0.158   | 0.356 | 0.166 | 0.136      | 0.306              |
|                | 256M  | 0.278       | 0.606 | 0.522        | 0.225   | 0.422 | 0.183 | 0.164      | 0.343              |
|                | 590M  | 0.291       | 0.634 | 0.479        | 0.281   | 0.475 | 0.206 | 0.152      | 0.360              |
|                | 1.3B  | 0.326       | 0.668 | 0.536        | 0.395   | 0.529 | 0.241 | 0.174      | 0.410              |
|                | 2.7B  | 0.382       | 0.697 | 0.543        | 0.487   | 0.590 | 0.267 | 0.224      | 0.456              |
|                | 6.7B  | 0.444       | 0.736 | 0.590        | 0.591   | 0.667 | 0.314 | 0.270      | 0.516              |
|                | 13B   | 0.514       | 0.768 | 0.674        | 0.655   | 0.743 | 0.398 | 0.318      | 0.581              |
| Cerebras muP   | 111M  | 0.268       | 0.581 | 0.520        | 0.146   | 0.368 | 0.175 | 0.124      | 0.312              |
|                | 256M  | 0.278       | 0.619 | 0.534        | 0.220   | 0.415 | 0.193 | 0.154      | 0.345              |
|                | 590M  | 0.298       | 0.652 | 0.515        | 0.301   | 0.479 | 0.206 | 0.174      | 0.375              |
|                | 1.3B  | 0.329       | 0.672 | 0.513        | 0.396   | 0.531 | 0.235 | 0.212      | 0.413              |
|                | 2.7B  | 0.382       | 0.704 | 0.560        | 0.510   | 0.595 | 0.267 | 0.210      | 0.461              |
| Pythia         | 70M   | 0.269       | 0.589 | 0.491        | 0.192   | 0.399 | 0.184 | 0.148      | 0.325              |
|                | 160M  | 0.292       | 0.631 | 0.515        | 0.329   | 0.469 | 0.205 | 0.164      | 0.372              |
|                | 410M  | 0.333       | 0.669 | 0.522        | 0.448   | 0.526 | 0.229 | 0.188      | 0.416              |
|                | 1B    | 0.374       | 0.709 | 0.562        | 0.514   | 0.596 | 0.265 | 0.206      | 0.461              |
|                | 1.4B  | 0.398       | 0.712 | 0.573        | 0.553   | 0.622 | 0.274 | 0.214      | 0.478              |
|                | 2.8B  | 0.448       | 0.738 | 0.621        | 0.629   | 0.673 | 0.328 | 0.254      | 0.527              |
|                | 6.9B  | 0.478       | 0.750 | 0.646        | 0.641   | 0.699 | 0.355 | 0.296      | 0.552              |
|                | 12B   | 0.506       | 0.759 | 0.662        | 0.673   | 0.731 | 0.383 | 0.322      | 0.577              |
| Pythia deduped | 70M   | 0.272       | 0.604 | 0.519        | 0.192   | 0.403 | 0.177 | 0.152      | 0.331              |
|                | 160M  | 0.294       | 0.639 | 0.507        | 0.309   | 0.472 | 0.215 | 0.178      | 0.373              |
|                | 410M  | 0.339       | 0.673 | 0.513        | 0.456   | 0.537 | 0.232 | 0.190      | 0.420              |
|                | 1B    | 0.384       | 0.710 | 0.552        | 0.529   | 0.588 | 0.259 | 0.226      | 0.464              |
|                | 1.4B  | 0.400       | 0.730 | 0.566        | 0.565   | 0.617 | 0.283 | 0.232      | 0.485              |
|                | 2.8B  | 0.463       | 0.758 | 0.609        | 0.637   | 0.681 | 0.327 | 0.282      | 0.537              |
|                | 6.9B  | 0.492       | 0.762 | 0.637        | 0.671   | 0.705 | 0.344 | 0.308      | 0.560              |
|                | 12B   | 0.516       | 0.765 | 0.678        | 0.696   | 0.728 | 0.386 | 0.326      | 0.585              |
| NeoX           | 20B   | 0.538       | 0.774 | 0.683        | 0.698   | 0.746 | 0.410 | 0.326      | 0.596              |
| GPTJ           | 6B    | 0.494       | 0.756 | 0.660        | 0.662   | 0.705 | 0.360 | 0.310      | 0.564              |
| OPT            | 125M  | 0.289       | 0.628 | 0.520        | 0.303   | 0.426 | 0.197 | 0.166      | 0.361              |
|                | 350M  | 0.321       | 0.647 | 0.521        | 0.384   | 0.464 | 0.208 | 0.184      | 0.390              |
|                | 1.3B  | 0.413       | 0.726 | 0.597        | 0.553   | 0.604 | 0.273 | 0.230      | 0.485              |
|                | 2.7B  | 0.458       | 0.749 | 0.616        | 0.603   | 0.651 | 0.305 | 0.276      | 0.523              |
|                | 6.7B  | 0.505       | 0.773 | 0.663        | 0.660   | 0.692 | 0.340 | 0.318      | 0.565              |
|                | 13B   | 0.524       | 0.763 | 0.684        | 0.678   | 0.714 | 0.358 | 0.306      | 0.575              |


<br><br>

## Uses and Limitations

### Intended Use
The primary intended use is to further research into large language models. These models can be used as a foundation model for NLP, applications, ethics, and alignment research. Our primary intended users are researchers who are working to improve LLMs and practitioners seeking reference implementations, training setups, hyperparameters, or pre-trained models. We release these models with a fully permissive Apache license for the community to use freely.

You may fine-tune and adapt Cerebras-GPT models for deployment via either Cerebras [Model Studio](https://www.cerebras.net/product-cloud/) or third-party libraries. Further safety-related testing and mitigations should be applied beore using the Cerebras-GPT model family in production downstream applications. 

Due to financial and compute budgets, Cerebras-GPT models were only trained and evaluated following the approaches described in the paper.

### Out of Scope Use
Cerebras-GPT models are trained on the Pile, with English language only, and are not suitable for machine translation tasks.

Cerebras-GPT models have not been tuned for human-facing dialog applications like chatbots and will not respond to prompts in a similar way to models that have received instruction tuning or reinforcement learning from human feedback (RLHF) like Flan-T5 or ChatGPT. Cerebras-GPT models can be tuned using those methods.

### Risk, Bias, Ethical Considerations
* **Data**: The Pile dataset has been thoroughly analyzed from various ethical standpoints such as toxicity analysis, gender bias, pejorative content, racially sensitive content etc. Please refer to Pile dataset references.
* **Human life**: The outputs from this model may or may not align with human values. The risk needs to be thoroughly investigated before deploying this model in a production environment where it can directly impact human life.
* **Risks and harms**: There can be distributional bias in the Pile dataset that can manifest in various forms in the downstream model deployment. There are other risks associated with large language models such as amplifying stereotypes, memorizing training data, or revealing private or secure information.
* **Mitigations**: Only mitigations in standard Pile dataset pre-processing were employed when pre-training Cerebras-GPT.

<br><br>


## Acknowledgements

We are thankful to all Cerebras engineers, past and present, that made this work possible.
