import torch
import torch.nn as nn
from torchvision import transforms as T
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from tqdm import tqdm
from typing import Optional, Callable, Tuple, List, Any
from dalle_pytorch import DALLE, OpenAIDiscreteVAE, DiscreteVAE
from dalle_pytorch.tokenizer import SimpleTokenizer
from torchvision.datasets.coco import CocoCaptions
from torch.utils.data import DataLoader
import torch.nn.functional as F
from einops import rearrange

# Change your input size here
input_image_size = 256

# Change your training batch size here
batch_size = 4

# Change your epoch here
epoch = 29

# Change your train image root path here
train_img_path = "./train2014/"

# Change your train annot json path here
train_annot_path = "./annotations/captions_train2014.json"

# Change your device ("cpu" or "cuda")
device = "cuda"

# Change your gpu device id (starts from 0 for first gpu if device is set to "cuda")
device_ids = [0, 1, 2, 3]

# Change your vae model save path here (ends with ".pth")
vae_save_path = "./vae.pth"

# Change your dalle model save path here (ends with ".pth")
dalle_save_path = "./dalle.pth"

transform = T.Compose([
    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    # T.Resize(input_image_size),
    T.RandomCrop(input_image_size, pad_if_needed=True),
    T.ToTensor()
])

tokenizer = SimpleTokenizer()

class JSONDataset(CocoCaptions):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        target = tokenizer.tokenize(target).squeeze(0)

        return image, target

train_data = JSONDataset(
    root=train_img_path,
    annFile=train_annot_path,
    transform=transform
)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# vae = DiscreteVAE(
#     image_size = 256,
#     num_layers = 3,
#     num_tokens = 8192,
#     codebook_dim = 1024,
#     hidden_dim = 64,
#     num_resnet_blocks = 1,
#     temperature = 0.9
# ).to(device)

vae = OpenAIDiscreteVAE().to(device)

# if os.path.exists(vae_save_path):
#     vae.load_state_dict(torch.load(vae_save_path))

# vae_parallel = nn.DataParallel(vae, device_ids=device_ids, output_device=[1], dim=0)

def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]

# opt = Adam(
#     get_trainable_params(vae_parallel),
#     lr = 3e-4,
#     # weight_decay=0.01,
#     # betas = (0.9, 0.999)
# )
# sched = ReduceLROnPlateau(
#     opt,
#     mode="min",
#     factor=0.5,
#     patience=10,
#     cooldown=10,
#     min_lr=1e-6,
#     verbose=True,
# )

# for curr_epoch in range(epoch):
#     print("Run training vae ...")
#     print(f"Epoch {curr_epoch+1} / {epoch}")

#     batch_idx = 0
    
#     for train_features, _ in tqdm(iter(train_loader)):
#         loss = vae_parallel(train_features, return_loss=True)

#         opt.zero_grad()
#         loss.mean().backward()
#         opt.step()
        
#         if batch_idx % 100 == 0:
#             torch.save(vae.state_dict(), vae_save_path)
#             print(f"average loss: {loss.mean().data}")
            
#         batch_idx += 1
        
#     sched.step(loss.mean())

# torch.save(vae.state_dict(), vae_save_path)

dalle = DALLE(
    dim = 1024,
    vae = vae,                                 # automatically infer (1) image sequence length and (2) number of image tokens
    num_text_tokens = tokenizer.vocab_size,    # vocab size for text
    text_seq_len = 256,                        # text sequence length
    depth = 22,                                # should aim to be 64
    heads = 16,                                # attention heads
    dim_head = 64,                             # attention head dimension
    attn_dropout = 0.1,                        # attention dropout
    ff_dropout = 0.1,                          # feedforward dropout
    # reversible = True,
    stable = True,
    optimize_for_inference = True
).to(device)

if os.path.exists(dalle_save_path):
    dalle.load_state_dict(torch.load(dalle_save_path))

dalle_parallel = nn.DataParallel(dalle, device_ids=device_ids, output_device=[1], dim=0)

opt = Adam(
    get_trainable_params(dalle_parallel),
    lr = 3e-4,
    # weight_decay=0.01,
    # betas = (0.9, 0.999)
)
sched = ReduceLROnPlateau(
    opt,
    mode="min",
    factor=0.5,
    patience=10,
    cooldown=10,
    min_lr=1e-6,
    verbose=True,
)

for curr_epoch in range(epoch):
    print("Run training dalle  ...")
    print(f"Epoch {curr_epoch+1} / {epoch}")

    batch_idx = 0
    
    for train_features, train_targets in tqdm(iter(train_loader)):
        if len(train_features) % len(device_ids) != 0:
            break

        loss = dalle_parallel(train_targets, train_features, return_loss=True)

        opt.zero_grad()
        loss.mean().backward()
        opt.step()
        
        if batch_idx % 100 == 0:
            torch.save(dalle.state_dict(), dalle_save_path)
            print(f"average loss: {loss.mean().data}")
            
        batch_idx += 1
        
    sched.step(loss.mean())

torch.save(dalle.state_dict(), dalle_save_path)