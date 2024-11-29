## Enriching Degradation Features for Fundus Image Enhancement via Multi-colour Dynamic Filter Network (ICONIP 2024)
---
Official Pytorch implementation
### Training
---
```python train.py --framework MFDyNet --batchsize 2 --load_index 0 --load_epoch 0 --epoch 150 --gan_loss_w 1 --d2g_lr 1 --gan_loss lsgan --resize 512```
### Testing
---
The training code automatically saves weights in the folder ```./myoutput``` and indexes each training. To select a specific network weight for testing, replace ```XX``` and ```YYY``` with the corresponding index and epoch numbers, respectively.  
```
python eval.py --framework MFDyNet --load_index XX --load_epoch YYY --dataset DATASETNAME dataset_path DATASETPATH --output_path OUTPUTPATH
```
### Citations
---
If you find our code useful, please consider citing our paper.

```
@InProceedings{MDA-Net,
          title = {Enriching Degradation Features for Fundus Image Enhancement via Multi-colour Dynamic Filter Network},
          booktitle = {International Conference on Neural Information Processing (ICONIP)},
          year = {2024},
          author = {Ruoyu Guo and Maurice Pagnucco and Yang Song}
}
```

