from coco import COCO

from PIL import Image
import requests
from io import BytesIO
import json
import yaml
import sys
import os
import argparse
from tqdm import tqdm


def str2bool(arg: str) -> bool:
    if arg in ['True', 't', 'T', '+']:
        return True
    if arg in ['False', 'f', 'F', '-']:
        return False
    return argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='COCO dataset transformation')
parser.add_argument('--download', '-d', type=str2bool,
                    default=False, required=False, metavar=('download'))
parser.add_argument('--max_amount', '-m', type=int,
                    default=sys.maxsize, required=False, metavar=('max_amount'))

config = yaml.safe_load(open('./config.yaml'))

coco_path = config['dataset']['coco_path']
images_dir = config['dataset']['images_dir']
annotations_path = config['dataset']['annotations_path']

coco = COCO(coco_path)


def transform_coco(max_amount: int, download: bool = False) -> None:
    """
    Transform COCO dataset into this repos data format
    Downloads images if specified
    """
    data = []

    bad_images = []
    cached_images = []

    pbar = tqdm(coco.anns)
    for i in pbar:
        # Load necessary metadata
        image_id = coco.anns[i]['image_id']
        caption = coco.anns[i]['caption']

        # print(f'Processing image_id {image_id}...   ', end=' ')
        pbar.set_description(f'Processing image_id {image_id}')
        url = coco.imgs[image_id]['url']
        file_name = coco.imgs[image_id]['file_name']

        # check if image was processed already
        if file_name in bad_images:
            continue

        if file_name in cached_images:
            data.append({
                'target': file_name,
                'prompt': caption,
                'source': '',
                'url': url
            })
            continue

        # process if wasnt
        if not os.path.isfile(f'{images_dir}/{file_name}'):
            response = requests.get(url)
            if not response.ok:
                bad_images.append(file_name)
                continue

            if download:
                img = Image.open(BytesIO(response.content))
                img.save(f'{images_dir}{file_name}')

            cached_images.append(file_name)

        data.append({
            'target': file_name,
            'prompt': caption,
            'source': '',
            'url': url
        })

        if len(data) > max_amount:
            break

    print('Saving data...')
    data = {"tags": data}

    with open(annotations_path, 'w') as outfile:
        json.dump(data, outfile)

    print(f'Done! {len(data["tags"])} image-caption pairs in total')


if __name__ == '__main__':
    args = parser.parse_args()

    transform_coco(download=args.download, max_amount=args.max_amount)
