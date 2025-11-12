from src.domain.file_handler import FileHandler

import boto3
import botocore.config
from io import BytesIO
from typing import Optional, List, Dict, Any
from requests import request
from PIL import Image


class S3FileHandler(FileHandler):
    def __init__(self) -> None:
        self.client_config = botocore.config.Config(
            max_pool_connections=32,
        )
        self.s3 = boto3.client(
            "s3", config=self.client_config
        )

    def get_object_list(self, bucket: str, key: str) -> List[Dict[str, Any]]:
        objects = self.s3.list_objects_v2(Bucket=bucket, Prefix=key)
        return objects["Contents"]
    
    def get_file_obj(self, bucket: str, key: str) -> BytesIO:
        obj = self.s3.get_object(Bucket=bucket, Key=key)
        return BytesIO(obj["Body"].read())

    def download_file_obj(self, bucket: str, key: str) -> BytesIO:
        file_obj = BytesIO()

        result = self.s3.download_fileobj(
            Bucket=bucket,
            Key=key,
            Fileobj=file_obj,
        )

        file_obj.seek(0) # Go to the start of the BytesIO object

        return file_obj

class CDNFileHandler(FileHandler):
    def __init__(self, cdn_url: str) -> None:
        self.cdn_url = cdn_url

    def get_file_obj(self, key: str) -> Optional[str]:
        return f"{self.cdn_url}{key}"

    def download_file_obj(self, key: str) -> Optional[BytesIO]:
        # download image from cdn url
        cdn_url = f"{self.cdn_url}{key}"    

        response = request("GET", cdn_url)
        if response.status_code != 200:
            # raise Exception(f"Failed to download image from {cdn_url}")
            return None

        image = Image.open(BytesIO(response.content)).convert("RGB")

        return image

    def save_image(self, image, path):
        image.save(path)