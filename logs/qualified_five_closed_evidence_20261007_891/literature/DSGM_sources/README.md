# Multi-Modal Object Re-Identification with Dual Semantic Guidance and Global-Local Mutual Modulation
Official PyTorch implementation of the paper:

**Multi-Modal Object Re-Identification with Dual Semantic Guidance and Global-Local Mutual Modulation**  
Authors: Weixiang Zhou, Xingguo Xu, Yuhao Wang, Cong Wang, Yang Yang, Zhixun Su, and Jinshan Pan

---

## **Quick Start** 🚀

### Datasets w/o Mask
- **RGBNT201**: [Google Drive](https://drive.google.com/drive/folders/1EscBadX-wMAT56_It5lXY-S3-b5nK1wH)  
- **RGBNT100**: [Baidu Pan](https://pan.baidu.com/s/1xqqh7N4Lctm3RcUdskG0Ug) (Code: `rjin`)  
- **MSVR310**: [Google Drive](https://drive.google.com/file/d/1IxI-fGiluPO_Ies6YjDHeTEuVYhFdYwD/view?usp=drive_link)

### Datasets w/ Mask
- **Three Datasets**: [Datasets](https://pan.baidu.com/s/1fZLqjZnBxFlMo-71TrbDBw) (Code: `rtns`)

### Pretrained Models
- **CLIP**: [Baidu Pan](https://pan.baidu.com/s/1YPhaL0YgpI-TQ_pSzXHRKw) (Code: `52fu`)

### Training
```bash
python train_net.py --config_file ./configs/RGBNT201/DSGM.yml
```

### Testing
```bash
python test_net.py --config_file ./configs/RGBNT201/DSGM.yml
```
### Training Example
- **RGBNT201**: [WEIGHT](https://pan.baidu.com/s/1kFYnBjoaDJBkEuk0q306LA)
- **CODE**: eicy

### Acknowledgement
Our code is based on [IDEA](https://github.com/924973292/IDEA)

