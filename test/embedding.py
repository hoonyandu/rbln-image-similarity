import argparse

from __init__ import *
from PIL import Image
from tqdm import tqdm

from src.infrastructure.model_loader import (RBLNAutoModelLoader,
                                             RBLNCLIPVisionModelLoader)
from src.presentations.embedding import generate_embeddings
from src.presentations.index import create_faiss_index


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="clip")
    parser.add_argument("--model-id", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--image-dir-path", type=str, default="data/image_dataset")

    parser.add_argument("--index-path", type=str, default="data/vector.index")

    return parser.parse_args()


def embedding_images(args):
    """
    Embed images and return embeddings
    Args:
        args: arguments

    Returns:
        embeddings: embeddings
    """
    if args.model_type == "clip":
        model_loader = RBLNCLIPVisionModelLoader(args.model_id)
    elif args.model_type == "qwen25vl":
        model_loader = RBLNAutoModelLoader(args.model_id)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")

    model = model_loader.load_model(args.model_path)
    processor = model_loader.load_processor()

    image_paths = [
        os.path.join(args.image_dir_path, image_path)
        for image_path in os.listdir(args.image_dir_path)
        if not image_path.startswith(".")
    ]
    embeddings = []

    for image_path in tqdm(image_paths):
        image = Image.open(image_path).convert("RGB")

        try:
            embedding = generate_embeddings(args.model_type, model, processor, image)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error generating embedding for {image_path}: {e}")
            continue

    return embeddings, image_paths


def main(args):
    embeddings, image_paths = embedding_images(args)
    index = create_faiss_index(embeddings, image_paths, args.index_path)


if __name__ == "__main__":
    args = get_args()
    main(args)
