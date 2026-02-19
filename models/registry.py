from .mpsenet.mpnet import MPNetModel
from .fastspeech2.fastspeech2 import FastSpeech2Model
from .conformer.conformer import ConformerModel
from .ast.ast import ASTModel
from .ecapa.ecapa import ECAPAModel

MODEL_METHODS = {
    "mpnet": MPNetModel,
    "fastspeech2": FastSpeech2Model,
    "conformer": ConformerModel,
    "ast": ASTModel,
    "ecapa": ECAPAModel
}

def init_model(model):
    return MODEL_METHODS[model]()