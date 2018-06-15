# Video Captioning

## TODO list
- [x] Preprocessing code [Textual part]
- [ ] Preprocessing code [Visual part]
- [ ] Model ensembling

## Dependencies ###

* Python 2.7
* Pytorch 0.2
* [Microsoft COCO Caption Evaluation](https://github.com/tylin/coco-caption)
* [CIDEr](https://github.com/plsang/cider)

(Check out the `coco-caption` and `cider` projects into your working directory)

## Data

Data can be downloaded [here](https://drive.google.com/drive/folders/1t65uYsDck6VV045GIaJXPIqL86vSGtyQ?usp=sharing) (643 MB). This folder contains: 
* input/msrvtt: annotatated captions (note that `val_videodatainfo.json` is a symbolic link to `train_videodatainfo.json`)
* output/feature: extracted features
* output/model/cst_best: model file and generated captions on test videos of our best run (CIDEr 54.2) 

## Getting started ###

Extract video features
  - Extracted features of ResNet, C3D, MFCC and Category embeddings are shared in the above link

Generate metadata

1. run `func_standalize_format`
2. run `func_preprocess_datainfo`
3. run `func_build_vocab`
4. run `func_create_sequencelabel`
5. run `func_convert_datainfo2cocofmt`
6. run `func_compute_ciderdf` # Pre-compute document frequency for CIDEr computation
7. run `func_compute_evalscores` # Pre-compute evaluation scores (BLEU_4, CIDEr, METEOR, ROUGE_L) for each caption


Please refer to the `opts.py` file for the set of available train/test options

### Training

```bash
# Train XE model
./train.sh 0 [GPUIDs]
```

```bash
# Train CST_GT_None/WXE model
./train.sh 1 [GPUIDs]
```

```bash
# Train CST_MS_Greedy model (using greedy baseline)
./train.sh 2 [GPUIDs]
```

```bash
# Train CST_MS_SCB model (using SCB baseline, where SCB is computed from GT captions)
./train.sh 3 [GPUIDs]
```



```bash
#Train CST_MS_SCB(*) model (using SCB baseline, where SCB is computed from model sampled captions)
./train.sh 4 [GPUIDs]
```

### Testing

```bash
./test.sh 0 [GPUIDs]
```

## Acknowledgements

* Torch implementation of [NeuralTalk2](https://github.com/karpathy/neuraltalk2)
* PyTorch implementation of Self-critical Sequence Training for Image Captioning [(SCST)](https://github.com/ruotianluo/self-critical.pytorch)
* PyTorch Team
* ["Consensus-based Sequence Training for Video Captioning" (Phan, Henter, Miyao, Satoh. 2017)](https://arxiv.org/abs/1712.09532).