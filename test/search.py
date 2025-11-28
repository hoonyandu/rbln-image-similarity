from __init__ import *
import argparse

from src.infrastructure.search_client import OpenSearchClient
from src.presentations.index import load_faiss_index, OpenSearchIndex, OpenSearchImageVectorMapping

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--response-timeout", type=int, default=1800)
    # OpenSearch는 기본적으로 SSL 사용 (docker-compose.yml 참고)
    parser.add_argument("--use-ssl", action="store_true", default=True)
    # self-signed 인증서를 사용하므로 verify_certs는 False
    parser.add_argument("--no-verify-certs", dest="verify_certs", action="store_false", default=False)
    parser.add_argument("--username", type=str, default="admin")
    parser.add_argument("--password", type=str, default="MyStr0ng!Password123")

    parser.add_argument("--index-path", type=str, default="data/vector.index")
    parser.add_argument("--image-prefix", type=str, default="https://img2.joongna.com/media/original/")
    parser.add_argument("--index-name", type=str, default="product-image-vector")
    
    return parser.parse_args()


def convert_faiss_to_documents(faiss_index, image_paths, image_prefix, batch_size=1000):
    """
    Convert faiss index to documents
    Args:
        faiss_index: faiss index
        image_paths: image paths
    Returns:
        dict[str, Any]: OpenSearch documents
    """
    total_vectors = faiss_index.ntotal
    documents = []
    
    for i in range(total_vectors):
        # vector = faiss_index.reconstruct(i)
        vector = faiss_index.index.reconstruct(i)

        image_path = image_paths[i]
        image_key = image_path.split("/")[-1].replace("-", "/")
        prod_media_seq = image_key.split("/")[0]
        image_url = f"{image_prefix}{"/".join(image_key.split('/')[1:])}"

        # OpenSearch document format
        doc = {
            "product_media_seq": prod_media_seq,
            "image_url": image_url,
            "image_vector": vector.tolist()
        }
        documents.append(doc)
    
        # 진행상황 출력
        if (i + 1) % batch_size == 0:
            print(f"Converted {i+1}/{total_vectors} ({(i+1) / total_vectors * 100:.2f}%)")

    return documents



def main(args):
    # Faiss index 로드
    index, image_paths = load_faiss_index(args.index_path)
    documents = convert_faiss_to_documents(index, image_paths, args.image_prefix)
    
    # OpenSearch client 생성
    client = OpenSearchClient(
        host=args.host, 
        port=args.port, 
        timeout=args.timeout, 
        response_timeout=args.response_timeout, 
        use_ssl=args.use_ssl, 
        verify_certs=args.verify_certs,
        username=args.username,
        password=args.password
    )
    print("Cluster Health:", client.get_cluster_info())

    # OpenSearch index 생성
    open_search_index = OpenSearchIndex(client)
    mapping_config = OpenSearchImageVectorMapping()
    # mapping_config는 객체이므로 .mapping 속성을 사용
    open_search_index.create_index(index_name=args.index_name, mapping=mapping_config.mapping)

    # OpenSearch index에 문서 인덱싱
    result = open_search_index.bulk_index_documents(index_name=args.index_name, documents=documents, id_field="product_media_seq")
    print(f"Bulk indexing completed. {result}")

    # OpenSearch index 통계 조회
    doc_count = open_search_index.get_index_statistics(args.index_name)

    # 유사 (이미지) 벡터 검색
    query_vector = index.index.reconstruct(100)
    similar_images = open_search_index.search_similar_vectors(args.index_name, query_vector)

    print("유사 (이미지) 벡터 검색 결과")
    for i, result in enumerate(similar_images, 1):
        print(f"{i}. Product: {result['product_media_seq']}, Score: {result['score']:.4f}")
        print(f"   URL: {result['image_url']}")
    


if __name__ == "__main__":
    args = get_args()
    main(args)