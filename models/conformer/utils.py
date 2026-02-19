import torch
import torchaudio
import torch.nn.functional as F

def load_featurizer():
    """
    Load the mel-spectrogram featurizer consistent with the model's training.
    """
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=80,
    )

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Decode the emission tensor to a string."""
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]

        # Join the decoded indices into a string and replace "-" with no space and "|" with spaces
        joined = "".join([self.labels[i] for i in indices])
        return joined.replace("-", "").replace("|", " ").strip().split()

def fix_length(tensor, target_len=734):
    D = tensor.size(1)
    if D > target_len:
        # Truncate
        tensor_fixed = tensor[:, :target_len, :]
    elif D < target_len:
        # Pad with zeros
        pad_len = target_len - D
        # pad format: (left_w, right_w, left_h, right_h, ...) for last two dims
        tensor_fixed = F.pad(tensor, (0, 0, 0, pad_len))  # pad along dim=1
    else:
        tensor_fixed = tensor
    return tensor_fixed