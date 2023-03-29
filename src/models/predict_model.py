# TODO: Add Gradio interface
from torchvision import transforms
from core.util import tensor2img
import torch
from pytorch_lightning import seed_everything
from PIL import Image
import utils

import yaml
config = yaml.safe_load(open('./config.yaml'))
model_config = config['training']['model_config_path']
model_config = yaml.safe_load(open(model_config))

seed_everything(config['misc']['seed'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#   Model args
paths: dict[str, str]           = config['models']

model_path: str                 = paths['load_path']
image_path: str                 = "who.cares"

#   Unet
unet_name: str                  = model_config['unet']['name']
unet_args: dict                 = model_config['unet']['unet_args']
scheduler_args: dict            = model_config['scheduler_args']

# TODO: It whould be a lot better to change this function to not initialize it for inference
#   like break it down to 'get_network' and 'init_network'
model = utils.get_network(unet_name, unet_args, scheduler_args)

state_dict = torch.load(model_path)
model.load_state_dict(state_dict, strict=False).to(device)
model.set_new_noise_schedule(device = device)
model.eval()

tfs = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Read input image
img_pillow = Image.open(image_path).convert('RGB')
cond_image = tfs(img_pillow)
cond_image = cond_image.to(device)

# Inference
with torch.no_grad():
    output, visuals = model.restoration(
        cond_image, 
        sample_num = 8
    )

# Save intermediates
output = output.detach().float().cpu()
for i in range(visuals.shape[0]):
    img = tensor2img(visuals[i].detach().float().cpu())
    Image.fromarray(img).save(f"./result/process_{i}.jpg")

# Save output 
img = tensor2img(output)
Image.fromarray(img).save("./result/output.jpg")
