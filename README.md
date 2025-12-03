# rbln-image-similarity

> An image similarity search system powered by RBLN (Rebellions NPU) acceleration.  
> Embed images into vectors and search for similar images using FAISS or OpenSearch.
> 
> 기간: 2025.11.12 ~ 2025.11.28 </br>
> 참여자: [hoonyandu](https://github.com/hoonyandu) </br>


## Structure

```
rbln-image-similarity/
    ├── src/
    │   ├── domain/                       # 도메인 인터페이스
    │   │   ├── file_handler.py
    │   │   ├── model_loader.py
    │   │   └── search_client.py
    │   ├── infrastructure/               # 인프라 구현체
    │   │   ├── file_handler.py
    │   │   ├── model_loader.py           # RBLN 모델 로더
    │   │   └── search_client.py          # OpenSearch 클라이언트
    │   ├── presentations/                # 비즈니스 로직
    │   │   ├── download.py
    │   │   ├── embedding.py              # 임베딩 생성 로직
    │   │   └── index.py                  # FAISS/OpenSearch 인덱스 관리
    │   └── utils/
    │       └── logger.py                 # 로깅 유틸리티
    ├── test/                             # 테스트 스크립트
    │   ├── embedding.py                  # 임베딩 테스트
    │   ├── search.py                     # 검색 테스트
    │   └── image_sim_search_faiss_clip.ipynb
    ├── scripts/
    │   └── env/
    │       └── create_dotenv.sh          # 환경변수 설정 스크립트
    ├── docker-compose.yml                # OpenSearch 클러스터 설정
    ├── requirements.txt
    └── README.md
```
</br>

## Environment
<pre>
<span style="color:blue">Python: </span><span style="color:green">3.10</span>
<span style="color:blue">OS: </span><span style="color:green">Ubuntu 22.04.5 LTS</span>
<span style="color:blue">Image: </span><span style="color:green">bai/atom-tensorflow-pytorch 3.10</span>
</pre>


### Supported Models

| 모델 | 설명 | 벡터 차원 |
|------|------|----------|
| `openai/clip-vit-base-patch32` | CLIP Vision 모델 (기본값) | 768 |
| `Qwen/Qwen2.5-VL` | Qwen2.5 Vision-Language 모델 | 3584 |

</br>

## Prerequisites

```bash
pip install -r requirements.txt
```

**Details:**

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `optimum-rbln` | 0.8.1 | RBLN NPU 가속 |
| `transformers` | 4.51.3 | 모델 로딩 |
| `torch` | 2.7.0 | 딥러닝 프레임워크 |
| `faiss-cpu` | 1.12.0 | 벡터 인덱싱 |
| `opensearch-py` | 3.0.0 | OpenSearch 클라이언트 |
| `pillow` | 11.0.0 | 이미지 처리 |

### 3. Run OpenSearch Cluster (Optional)

To use OpenSearch for large-scale vector search:

```bash
docker-compose up -d
```

**OpenSearch Connection:**

| 항목 | 값 |
|------|-----|
| Host | `localhost` |
| Port | `9200` |
| Dashboard | `http://localhost:5601` |
| Username | `admin` |
| Password | `MyStr0ng!Password123` |

---

## Usage

### 1. Embed Image Vectors

Embeds all images from the image directory and creates a FAISS index.

```bash
python test/embedding.py \
    --model-type clip \
    --model-id openai/clip-vit-base-patch32 \
    --image-dir-path data/image_dataset \
    --index-path data/vector.index
```

### 2. OpenSearch Vector Search

Migrates the FAISS index to OpenSearch and searches for similar images.

```bash
python test/search.py \
    --host localhost \
    --port 9200 \
    --use-ssl \
    --username admin \
    --password "MyStr0ng!Password123" \
    --index-path data/vector.index \
    --image-prefix "https://your-cdn.com/images/" \
    --index-name product-image-vector
```

### 3. Python Example

```python
from PIL import Image
from src.infrastructure.model_loader import RBLNCLIPVisionModelLoader
from src.presentations.embedding import generate_embeddings
from src.presentations.index import load_faiss_index

# 모델 로드
model_loader = RBLNCLIPVisionModelLoader("openai/clip-vit-base-patch32")
model = model_loader.load_model()
processor = model_loader.load_processor()

# 이미지 임베딩 생성
image = Image.open("path/to/image.jpg").convert("RGB")
embedding = generate_embeddings("clip", model, processor, image)

# FAISS 인덱스 로드 및 검색
index, image_paths = load_faiss_index("data/vector.index")
distances, indices = index.search(embedding.reshape(1, -1), k=10)

# 유사 이미지 출력
for i, idx in enumerate(indices[0]):
    print(f"Rank {i+1}: {image_paths[idx]} (distance: {distances[0][i]:.4f})")
```