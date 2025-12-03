from optimum.rbln import RBLNAutoModelForVision2Seq, RBLNCLIPVisionModel
from transformers import AutoProcessor

from src.domain.model_loader import ModelLoader


class RBLNCLIPVisionModelLoader(ModelLoader):
    """
    RBLNCLIPVisionModel 로더
    Args:
        model_id: 모델 ID

    Returns:
        RBLNCLIPVisionModel
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def load_model(self, model_path: str = None) -> RBLNCLIPVisionModel:
        if model_path:
            model = RBLNCLIPVisionModel.from_pretrained(model_path)
        else:
            model = RBLNCLIPVisionModel.from_pretrained(self.model_id, export=True)
            model.save_pretrained(self.model_id)

        return model

    def load_processor(self) -> AutoProcessor:
        processor = AutoProcessor.from_pretrained(self.model_id)

        return processor


class RBLNAutoModelLoader(ModelLoader):
    """
    RBLNAutoModelForVision2Seq 로더
    Args:
        model_id: 모델 ID

    Returns:
        RBLNAutoModelForVision2Seq
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def load_model(self, model_path: str = None) -> RBLNAutoModelForVision2Seq:
        if model_path:
            model = RBLNAutoModelForVision2Seq.from_model(model_path)
        else:
            # model_id: "Qwen/Qwen2.5-VL-7B-Instruct"
            # https://github.com/rebellions-sw/rbln-model-zoo/blob/main/huggingface/transformers/image-text-to-text/qwen2.5-vl/qwen2.5-vl-7b/compile.py
            model = RBLNAutoModelForVision2Seq.from_pretrained(
                self.model_id, 
                export=True,
                rbln_create_runtimes=False,
                rbln_config={
                    "visual": {
                        "max_seq_lens": 6400,
                        "device": 0,
                    },
                    "max_seq_len": 32_768,
                },
            )
            model.save_pretrained(self.model_id)

        return model

    def load_processor(self) -> AutoProcessor:
        processor = AutoProcessor.from_pretrained(self.model_id)

        return processor
