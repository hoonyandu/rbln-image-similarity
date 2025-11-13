import argparse

from __init__ import *
from tqdm import tqdm

from src.presentations.download import (download_csv_from_s3,
                                        download_image_from_cdn)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=str)
    parser.add_argument("--key", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--cdn_url", type=str)
    parser.add_argument("--save_dir", type=str)

    return parser.parse_args()


def main(args):
    df = download_csv_from_s3(
        bucket=args.bucket, key=args.key, output_path=args.output_path
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for idx, row in tqdm(df.iterrows()):
        product_seq = row["product_seq"]
        image_key = row["thumbnail_url"]
        image_path = f'{product_seq}-{image_key.replace("/media/original/", "").replace("/", "-")}'

        try:
            download_image_from_cdn(args.cdn_url, args.save_dir, image_key, image_path)

        except Exception as e:
            print(f"Error downloading image: {e}")
            print(f"Image idx, key: {idx}, {image_key}")
            continue


if __name__ == "__main__":
    args = get_args()
    main(args)
