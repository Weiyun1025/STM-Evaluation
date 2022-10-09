import os
import argparse


def parse(base_dir):
    for dirname in ('train', 'val'):
        meta = []
        num_labels = 0
        dirname = os.path.join(base_dir, dirname)
        print(dirname)
        for idx, label in enumerate(os.listdir(dirname)):
            image_dir = os.path.join(dirname, label)
            for image in os.listdir(image_dir):
                meta.append(f'{os.path.join(label, image)} {idx}\n')

            num_labels = idx

        meta_path = os.path.join(base_dir, 'meta', f'{dirname}.txt')
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        print(meta_path)
        with open(meta_path, 'w', encoding='utf-8') as file:
            file.writelines(meta)

        print(f'{dirname} num_labels: {num_labels + 1}')
        print(f'{dirname} samples: {len(meta)}')
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='/root/ImageNet')
    args = parser.parse_args()

    parse(args.base_dir)


if __name__ == '__main__':
    main()
