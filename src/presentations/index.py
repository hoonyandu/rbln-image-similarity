import faiss
import numpy as np

def create_faiss_index(embeddings, image_paths, index_path):
    dimension = len(embeddings[0])
    # L2 distance index
    index = faiss.IndexFlatL2(dimension)
    # ID map index
    index = faiss.IndexIDMap(index)

    vectors = np.array(embeddings).astype("float32")

    # Add vectors to the index with IDs
    index.add_with_ids(vectors, np.arange(len(embeddings)))

    # Save the index
    faiss.write_index(index, index_path)

    # Save image paths
    with open(index_path + ".paths", "w") as f:
        for image_path in image_paths:
            f.write(image_path + "\n")

    return index

def load_faiss_index(index_path):
    index = faiss.read_index(index_path)
    with open(index_path + ".paths", "r") as f:
        image_paths = [line.strip() for line in f]
    print(f"Index loaded from {index_path}")

    return index, image_paths