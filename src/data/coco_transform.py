# Cloned from https://github.com/junjiedong/Image_Captioning_MSCOCO
import json
import yaml

config = yaml.safe_load(open('./config.yaml'))

coco_path = config['dataset']['coco_path']
annotations_path = config['dataset']['annotations_path']


def main() -> None:
    """
    Transform COCO 2017 dataset into this repos data format
    """

    # Load data
    with open(coco_path) as f:
        data = json.load(f)

    itoa = {}
    for a in data['annotations']:
        imgid = a['image_id']
        if imgid not in itoa:
            itoa[imgid] = []
        itoa[imgid].append(a)

    # Create the json blob
    out = []
    for img in data['images']:
        imgid = img['id']
        jimg = {}
        jimg['file_path'] = img['file_name']
        jimg['id'] = imgid
        # For Palette we are not using any captions
        # jimg['captions'] = [a['caption'] for a in itoa[imgid]]
        jimg['captions'] = ['No caption']
        jimg['url'] = img['coco_url']

        out.append(jimg)

    # Create image-caption pairs
    pairs = []
    for entry in out:
        for caption in entry['captions']:
            pairs.append({
                'target': entry['file_path'],
                'prompt': caption,
                'source': '',
                'url': entry['url']
            })

    # Save
    pairs = {
        'dataset_name': 'COCO Captions train 2017',
        'tags': pairs
    }
    json.dump(pairs, open(annotations_path, 'w'))

    print(f'Done! {len(pairs["tags"])} image-caption pairs in total')


if __name__ == '__main__':
    main()
