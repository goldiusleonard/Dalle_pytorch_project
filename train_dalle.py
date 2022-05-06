import torch
from torchvision import transforms as T
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import os
from tqdm import tqdm
from dalle_pytorch import DiscreteVAE, DALLE
from dalle_pytorch.tokenizer import SimpleTokenizer
from torchvision.datasets.coco import CocoCaptions

# Change your input size here
input_image_size = 256

# Change your batch size here
batch_size = 1

# Change your epoch here
epoch = 5

# Change your train image root path here
train_img_path = "./train2014/"

# Change your train annot json path here
train_annot_path = "./annotations/captions_train2014.json"

# Change your device ("cpu" or "cuda")
device = "cuda"

# Change your VAE save path here (ends with ".pth")
vae_save_path = "./vae.pth"

# Change your dalle model save path here (ends with ".pth")
dalle_save_path = "./dalle.pth"

# Change the test result image save path (should be a directory or folder)
test_img_save_path = "./result"

if not os.path.exists(test_img_save_path):
    os.makedirs(test_img_save_path)

transform = T.Compose([
    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    T.Resize(input_image_size),
    T.CenterCrop(input_image_size),
    T.ToTensor()
])

train_data = CocoCaptions(
    root=train_img_path,
    annFile=train_annot_path,
    transform=transform
)

vae = DiscreteVAE(
    image_size = 256,
    num_layers = 3,           # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
    num_tokens = 8192,        # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
    codebook_dim = 512,       # codebook dimension
    hidden_dim = 64,          # hidden dimension
    num_resnet_blocks = 1,    # number of resnet blocks
    temperature = 0.9,        # gumbel softmax temperature, the lower this is, the harder the discretization
    straight_through = False, # straight-through for gumbel softmax. unclear if it is better one way or the other
).to(device)

train_size = len(train_data)
idx_list = range(0, train_size, batch_size)

tokenizer = SimpleTokenizer()
opt = Adam(
    vae.parameters(),
    lr = 3e-4,
    weight_decay=0.01,
    betas = (0.9, 0.999)
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
    print("Run training discrete vae ...")
    print(f"Epoch {curr_epoch+1} / {epoch}")
    
    for batch_idx in tqdm(idx_list):
        if (batch_idx + batch_size) > train_size - 1:
            iter_idx = range(batch_idx, train_size, 1)
        else:
            iter_idx = range(batch_idx, batch_idx+batch_size, 1)

        image_list = []
        
        for curr_idx in iter_idx:
            image, _ = train_data[curr_idx]
            image = image.unsqueeze(0)

            image_list.append(image)

        images = torch.cat(image_list, dim=0).type(torch.FloatTensor).to(device)

        opt.zero_grad()
        loss = vae(images, return_loss = True)
        loss.backward()
        opt.step()

        if batch_idx != 0 and batch_idx % 100 == 0:
            torch.save(vae.state_dict(), vae_save_path)
            sched.step(loss)
        
        if batch_idx % 1000 == 0:
            print(f"loss: {loss.data}")

torch.save(vae.state_dict(), vae_save_path)

tokenizer = SimpleTokenizer()

dalle = DALLE(
    dim = 1024,
    vae = vae,                                  # automatically infer (1) image sequence length and (2) number of image tokens
    num_text_tokens = 10000,                    # vocab size for text
    text_seq_len = 256,                         # text sequence length
    depth = 1,                                  # should aim to be 64
    heads = 16,                                 # attention heads
    dim_head = 64,                              # attention head dimension
    attn_dropout = 0.1,                         # attention dropout
    ff_dropout = 0.1                            # feedforward dropout
).to(device)

train_size = len(train_data)
idx_list = range(0, train_size, batch_size)

opt = Adam(
    dalle.parameters(),
    lr = 3e-4,
    weight_decay=0.01,
    betas = (0.9, 0.999)
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
    print("Run training dalle ...")
    print(f"Epoch {curr_epoch+1} / {epoch}")
    
    for batch_idx in tqdm(idx_list):
        if (batch_idx + batch_size) > train_size - 1:
            iter_idx = range(batch_idx, train_size, 1)
        else:
            iter_idx = range(batch_idx, batch_idx+batch_size, 1)

        image_list = []
        text_list = []
        
        for curr_idx in iter_idx:
            image, target = train_data[curr_idx]
            image = image.unsqueeze(0)
            text = tokenizer.tokenize(target)

            text_size = len(text)
            for i in range(text_size):
                image_list.append(image)
            
            text_list.append(text)

        text = torch.cat(text_list, dim=0).to(device)
        image = torch.cat(image_list, dim=0).to(device)

        opt.zero_grad()
        loss = dalle(text, image, return_loss = True)
        loss.backward()
        opt.step()

        if batch_idx != 0 and batch_idx % 100 == 0:
            torch.save(dalle.state_dict(), dalle_save_path)
            sched.step(loss)
        
        if batch_idx % 1000 == 0:
            print(f"loss: {loss.data}")

torch.save(dalle.state_dict(), dalle_save_path)

test_inputs = ['Closeup of bins of food that include broccoli and bread.'] # text input for the model (can be more than one)

text = tokenizer.tokenize(test_inputs).to(device)

test_img_tensors = dalle.generate_images(text)

for test_idx, test_img_tensor in enumerate(test_img_tensors):
    test_img = T.ToPILImage()(test_img_tensor)
    test_save_path = os.path.join(test_img_save_path, f"{test_inputs[test_idx]}.jpg")
    test_img.save(Path(test_save_path))