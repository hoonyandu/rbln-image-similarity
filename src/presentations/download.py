import os

import pandas as pd

from src.infrastructure.file_handler import CDNFileHandler, S3FileHandler


def download_csv_from_s3(bucket: str, key: str, output_path: str) -> pd.DataFrame:
    """
    Meta data CSV 파일 다운로드

    Args:
        bucket: S3 bucket name
        key: S3 key name
        output_path: 출력 파일 이름

    Returns:
        pd.DataFrame: 다운로드 된 데이터프레임
    """
    df = pd.DataFrame()

    file_handler = S3FileHandler()
    obj_list = file_handler.get_object_list(bucket, key)
    obj_list = [obj["Key"] for obj in obj_list if obj["Key"].endswith(".csv")]

    for obj in obj_list:
        print(f"Processing: {obj}")
        file_obj = file_handler.download_file_obj(bucket, obj)
        obj_df = pd.read_csv(file_obj)
        df = pd.concat([obj_df, df])

    df.to_csv(output_path, index=False)

    return df


def download_image_from_cdn(
    cdn_url: str, save_dir: str, image_key: str, image_path: str
) -> None:
    """
    CDN 이미지 다운로드

    Args:
        cdn_url: CDN URL
        save_dir: 저장 디렉토리
        image_key: 이미지 키
        image_path: 이미지 저장 경로

    Returns:
        None
    """
    file_handler = CDNFileHandler(cdn_url)
    image = file_handler.download_file_obj(key=image_key)
    file_handler.save_image(image, os.path.join(save_dir, image_path))
