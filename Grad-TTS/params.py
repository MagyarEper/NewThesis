# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from model.utils import fix_len_compatibility


# data parameters
train_filelist_path = 'resources/filelists/train_textplit.txt'
valid_filelist_path = 'resources/filelists/valid_textsplit.txt'
test_filelist_path = 'resources/filelists/libri-tts/test.txt'
cmudict_path = 'resources/cmu_dictionary'
add_blank = True
n_feats = 80
n_spks = 39  # Hungarian Dysarthria Database: 39 speakers
spk_emb_dim = 64
n_feats = 80
n_fft = 1024
sample_rate = 16000  # 16kHz for our dataset
hop_length = 256
win_length = 1024
f_min = 0
f_max = 8000

# encoder parameters (optimized for 16GB VRAM)
n_enc_channels = 128  # 192 -> 128 (moderate size)
filter_channels = 512  # 768 -> 512 (moderate)
filter_channels_dp = 192  # 256 -> 192 (moderate)
n_enc_layers = 5  # 6 -> 5 (one less layer)
enc_kernel = 3
enc_dropout = 0.1
n_heads = 2
window_size = 4

# decoder parameters (optimized for 16GB VRAM)
dec_dim = 48  # 64 -> 48 (moderate size)
beta_min = 0.05
beta_max = 20.0
pe_scale = 1000  # 1 for `grad-tts-old.pt` checkpoint

# training parameters (optimized for 16GB VRAM)
log_dir = 'logs/hungarian_dysarthria_v2'
test_size = 4  # back to 4 test samples
n_epochs = 200  # converges ~150-200 epochs
batch_size = 12  # 8 -> 12 (good balance for 16GB)
learning_rate = 1e-4
seed = 37
save_every = 10  # save checkpoints every 10 epochs for better analysis
out_size = fix_len_compatibility(2*16000//256)  # adjusted for 16kHz
