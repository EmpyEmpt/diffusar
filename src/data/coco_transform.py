from coco import COCO

from PIL import Image
import requests
from io import BytesIO
import json
import yaml
import sys
import os

config = yaml.safe_load(open('./config.yaml'))

coco_path = config['dataset']['coco_path']
images_dir = config['dataset']['images_dir']
annotations_path = config['dataset']['annotations_path']

coco = COCO(coco_path)


def transform_coco(download=True):
    """
    Transform COCO dataset into this repos data format
    Downloads images if specified
    """
    data = []

    for i in coco.anns:
        image_id = coco.anns[i]['image_id']
        caption = coco.anns[i]['caption']

        print(f'Processing image_id {image_id}...   ', end=' ')
        # image = coco.imgs[image_id]
        url = coco.imgs[image_id]['url']
        file_name = coco.imgs[image_id]['file_name']

        if not os.path.isfile(f'{images_dir}/{file_name}'):
            response = requests.get(url)
            if not response.ok:
                print("Image is not accessible :(")
                continue

            if download:
                img = Image.open(BytesIO(response.content))
                img.save(f'{images_dir}{file_name}')

        print('COMPLETE')
        data.append({
            'target': file_name,
            'prompt': caption,
            'source': '',
            'url': url
        })

    print('Sacing data...')
    data = {"tags": data}

    with open(annotations_path, 'w') as outfile:
        json.dump(data, outfile)
    print(f'Done! {len(data)} image-caption pairs in total')


if __name__ == '__main__':
    download = False
    if len(sys.argv) > 1:
        download = True
    transform_coco(download=download)
