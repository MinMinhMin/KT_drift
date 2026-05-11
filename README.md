### Dataset, embedding, checkpoint (best weight), result (clustering) link: https://drive.google.com/drive/folders/11J0n2gF2zM2ViTFUqYKLPJVlDOVQOcvt?usp=sharing
```bash
├── dataset/
├── checkpoints/
├──clustering/
còn lại là các file .py, .ipynb
```

### Tiền xử lý dataset (preprocess.py (đặc thù cho XES3G5M))
```bash
python preprocess.py
```

### Train learnable embedding (id embedding, tree embedding) - ví dụ cho ver hiện tại (còn các thẻ hyper param khác đề cập trong parser, ở đây dùng default)
```bash
    python train_id_embedding.py \
         --csv dataset/processed/XES3G5M/processed.csv \
         --output dataset/processed/XES3G5M/excercices_embedding\
         --embed_dim 16 \
         --r_embed_dim 4 \
         --hidden_dim 64 \
```

```bash
    python train_tree_embedding.py \
         --csv dataset/processed/XES3G5M/processed.csv \
         --question_dict dataset/processed/XES3G5M/question_dict.pkl \
         --kc_dict dataset/processed/XES3G5M/kc_dict.pkl \
         --output dataset/processed/XES3G5M/excercices_embedding/tree_embedding\
         --emb_dim 16 \
         --n_levels 4 \
```

### Training representaion model, ví dụ cho ver hiện tại (còn các thẻ hyper param khác đề cập trong parser, ở đây dùng default)
```bash
  python training.py \
     --csv_path dataset/processed/XES3G5M/processed.csv \
     --ablation_mode no_text \
     --model autoencoder \
```
### Visualize clustering chạy find_k.py để tìm best K và visualize bằng notebook vis.ipynb
```bash
python find_k.py \
         --checkpoint checkpoints/autoencoder-no_text/best.pt \
         --output clustering \
         --index 0 \
         --ablation no_text 
```
### Chức năng các file .py khác 
- dataset.py: dataloader cho training representation
- get_feature.py: embedding manager dùng cho dataset.py
- verify_tree_embedding.py: kiểm tra tính chính xác của tree embedding
- autoencoder.py: mô hình representation
- test_build_dataset.py: kiểm tra hoạt động của dataset
- usage.md: XES3G5M Dataset Pipeline

