# GCN-LFR
This repository is the official PyTorch implementation of "Not All Low-Pass Filters are Robust in Graph Convolutional Networks".

Heng Chang, Yu Rong, Tingyang Xu, Yatao Bian, Shiji Zhou, Xin Wang, Junzhou Huang, Wenwu Zhu, [Not All Low-Pass Filters are Robust in Graph Convolutional Networks](https://openreview.net/pdf?id=bDdfxLQITtu), NeurIPS 2021.

## Requirements
The script has been tested running under Python 3.6.9, with the following packages installed (along with their dependencies):
* pytorch (tested on 1.7.1)
* torch_geometric (tested on 1.6.3)
* scipy (tested on 1.5.4)
* numpy (tested on 1.19.5)
* networkx (tested on 2.5.1)
* sklearn (tested on 0.24.2)
* deeprobust (tested on 0.1.1)

## Datasets
The datasets are from PyG, which can be referred to the [docs](https://pytorch-geometric.readthedocs.io/en/latest/notes/data_cheatsheet.html).

## Run
- For the defense experiment on Cora dataset, one-edge targeted attack under Nettack (default setting):
```bash
python LFR_test.py
```

## Acknowledgement
Part of this implementation is modified from [DeepRobust](https://github.com/DSE-MSU/DeepRobust/tree/master/deeprobust/graph), and we sincerely thank them for their contributions.

## Reference
- If you find ``GCN-LFR`` useful in your research, please cite the following in your manuscript:

```
@article{chang2021not,
  title={Not All Low-Pass Filters are Robust in Graph Convolutional Networks},
  author={Chang, Heng and Rong, Yu and Xu, Tingyang and Bian, Yatao and Zhou, Shiji and Wang, Xin and Huang, Junzhou and Zhu, Wenwu},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```


