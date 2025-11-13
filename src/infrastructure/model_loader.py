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
            model = RBLNAutoModelForVision2Seq.from_pretrained(
                self.model_id, export=True
            )
            model.save_pretrained(self.model_id)

        return model

    def load_processor(self) -> AutoProcessor:
        processor = AutoProcessor.from_pretrained(self.model_id)

        return processor
