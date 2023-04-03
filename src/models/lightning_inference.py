# TODO: Add Gradio interface
import yaml
import lightning_model as lm
import lightning_data as ld
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from PIL import Image

config = yaml.safe_load(open('./config.yaml'))
model_config = yaml.safe_load(open(config['training']['model_config_path']))

# Image should be in ./test_images/data/
images_path: str = "./test_images/"

model = lm.Diffusar(
    unet_name=model_config['unet']['name'],
    unet_args=model_config['unet']['unet_args'],
    scheduler_args=model_config['scheduler_args'],
    optimizer_name=model_config['optimizer']['name'],
    optimizer_args=model_config['optimizer']['optimizer_args'],
    loss_name=model_config['loss']['loss_name'],
    ema_scheduler=None
)
model.load_checkpoint('./models/checkpoints/test.ckpt')


if __name__ == '__main__':
    seed_everything(config['misc']['seed'])

    data_module = ld.InferenceData(images_path, 1)
    trainer = Trainer(accelerator="cpu")

    output = trainer.predict(
        model, datamodule=data_module, return_predictions=True)

    img = output[0][0]
    print(output)
