import faiss
import numpy as np
import re
from typing import Dict, Any, Optional, List
from tqdm import tqdm
import time

from src.infrastructure.search_client import OpenSearchClient
from src.utils.logger import get_logger


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


class OpenSearchIndex:
    def __init__(self, client: OpenSearchClient) -> None:
        self.client = client.client
        self.logger = get_logger(__name__)

    def create_index(
        self, index_name: str, mapping: Dict[str, Any], overwrite: bool = False
    ) -> None:
        """
        Create OpenSearch index
        Args:
            index_name: index name
            mapping: index mapping
            overwrite: overwrite
        Returns:
            response: response
        """
        self._validate_index_name(index_name)

        try:
            exists = self.client.indices.exists(index=index_name)
        except Exception as e:
            self.logger.error(f"Error creating index: {e}")
            raise e

        response = {
            "acknowledged": False,
            "message": "Index does not exist",
        }
        if exists:
            if overwrite:
                self.logger.warning(
                    f"Index {index_name} already exists. Overwriting..."
                )
                self.client.indices.delete(index=index_name)
            else:
                self.logger.info(
                    f"Index {index_name} already exists. Use overwrite=True to overwrite."
                )
                response["message"] = "Index already exists"

        else:
            self.logger.info(f"Index {index_name} does not exist. Creating...")
            response = self.client.indices.create(index=index_name, body=mapping)

        return response

    def _validate_index_name(self, index_name: str) -> None:
        """
        Validate OpenSearch index name
        Args:
            index_name: index name
        Returns:
            None
        """
        # 1. Index name is required
        if not index_name:
            raise ValueError("Index name is required")

        # 2. Index name must be lowercase
        if index_name != index_name.lower():
            raise ValueError("Index name must be lowercase")

        # 3. Index name cannot start or end with a dot
        if index_name.startswith(".") or index_name.endswith("."):
            raise ValueError("Index name cannot start or end with a dot")

        # 4. Index name must contain only lowercase letters, numbers, underscores, and hyphens
        if not re.match(r"^[a-z0-9_+-]+$", index_name):
            raise ValueError(
                "Index name must contain only lowercase letters, numbers, underscores, and hyphens"
            )

        # 5. Index name must be less than 255 characters
        if len(index_name) > 255:
            raise ValueError("Index name must be less than 255 characters")

    def bulk_index_documents(
        self,
        index_name,
        documents,
        id_field,
        batch_size: int = 500,
        max_retries: int = 3,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Bulk index documents
        Args:
            index_name: index name (str)
            documents: documents (list of dict)
            id_field: id field (str)
        Returns:
            response: response
        """

        total_docs = len(documents)
        self.logger.info(f"Bulk indexing {total_docs} documents to {index_name}")

        success_count = 0
        failed_count = 0
        failed_docs = []

        batches = [
            documents[i : i + batch_size] for i in range(0, total_docs, batch_size)
        ]

        iterator = (
            tqdm(batches, desc="Bulk indexing", disable=not show_progress)
            if show_progress
            else batches
        )

        for batch in iterator:
            actions = []
            for doc in batch:
                action = {
                    "index": {
                        "_index": index_name,
                        "_id": doc[id_field],
                    }
                }
                actions.append(action)
                actions.append(doc)

            # 공통 재시도 로직 사용
            result = self._execute_bulk_index(actions, "index", max_retries)

            success_count += result["success"]
            failed_count += result["failed"]
            failed_docs.extend(result["failed_items"])

        result = {
            "success": success_count,
            "failed": failed_count,
            "total": total_docs,
            "failed_docs": failed_docs[:100] if failed_docs else [],
        }

        self.logger.info(
            f"Bulk indexing completed. Success: {success_count}, Failed: {failed_count}, Total: {total_docs}"
        )
        return result

    def _execute_bulk_index(
        self,
        actions,
        operation_type,
        max_retries: int = 3,
        expected_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """

        재시도 로직을 포함한 bulk index 실행 함수

        Args:
            actions: actions (list of dict)
            operation_type: 작업 타입 ("index", "update", "delete")
            max_retries: 최대 재시도 횟수 (int)
            expected_count: 예상 작업 수 (int, None이면 len(actions) // 2)
        Returns:
            response: Dict[str, Any]
        """
        retry_count = 0

        # expected_count가 지정되지 않으면 actions 길이의 절반으로 계산 (index/update의 경우)
        if expected_count is None:
            expected_count = len(actions) // 2

        while retry_count < max_retries:
            try:
                response = self.client.bulk(body=actions, request_timeout=60)

                success_count = 0
                failed_count = 0
                failed_items = []

                if response.get("errors"):
                    results = [
                        item.get(operation_type, {})
                        for item in response["items"]
                        if operation_type in item
                    ]
                    failed_items = [
                        {"id": r.get("_id"), "error": r.get("error")}
                        for r in results
                        if r.get("status", 0) >= 400
                    ]
                    failed_count = len(failed_items)
                    success_count = len(results) - failed_count
                else:
                    success_count = expected_count

                bulk_result = {
                    "success": success_count,
                    "failed": failed_count,
                    "failed_items": failed_items,
                }
                break  # 성공 시 반복문 탈출

            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    self.logger.warning(
                        f"Bulk operation failed (attempt {retry_count}/{max_retries}): {e}"
                    )
                    time.sleep(2**retry_count)
                    continue

                self.logger.error(
                    f"Bulk operation failed after {max_retries} retries: {e}"
                )
                bulk_result = {
                    "success": 0,
                    "failed": expected_count,
                    "failed_items": [],
                }
                break  # 실패 시 반복문 탈출

        return bulk_result  # 성공 또는 실패 결과 반환

    def get_index_statistics(self, index_name: str) -> int:
        """
        Get index statistics
        Args:
            index_name: index name
        Returns:
            int: Document count
        """
        stats = self.client.indices.stats(index=index_name)
        doc_count = stats["indices"][index_name]["total"]["docs"]["count"]

        self.logger.info(f"Index {index_name} has {doc_count} documents")

        return doc_count

    def search_similar_vectors(
        self, index_name: str, query_vector: List[float], k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search similar vectors
        Args:
            index_name: index name
            query_vector: query vector
            k: number of results
        Returns:
            List[Dict[str, Any]]: List of search results
        """
        query_vector = (
            query_vector.tolist()
            if isinstance(query_vector, np.ndarray)
            else query_vector
        )
        results = []

        search_query = {
            "size": k,
            "query": {
                "knn": {
                    "image_vector": {
                        "vector": query_vector,
                        "k": k,
                    }
                }
            },
        }

        response = self.client.search(index=index_name, body=search_query)
        for hit in response["hits"]["hits"]:
            results.append(
                {
                    "product_media_seq": hit["_source"]["product_media_seq"],
                    "image_url": hit["_source"]["image_url"],
                    "score": hit["_score"],
                }
            )

        return results


class OpenSearchImageVectorMapping:
    """
    OpenSearch image vector mapping
    Args:
        None
    Returns:
        mapping: mapping
    """

    @property
    def mapping(self) -> Dict[str, Any]:
        """
        image-vector 인덱스의 매핑을 반환합니다.

        상품 이미지 임베딩 벡터를 저장하는 인덱스입니다.
        - KNN 검색 활성화
        - HNSW 알고리즘 사용 (Lucene 엔진)
        - Cosine similarity 사용

        Returns:
            mapping: Dict[str, Any]
        """

        return {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "refresh_interval": "30s",
                },
            },
            "mappings": {
                "properties": {
                    "product_media_seq": {
                        "type": "long",
                    },
                    "image_url": {
                        "type": "keyword",
                    },
                    "image_vector": {
                        "type": "knn_vector",
                        "dimension": 768,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "lucene",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 16,
                            },
                        },
                    },
                }
            },
        }
