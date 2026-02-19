from sklearn import metrics
import numpy as np
from operator import itemgetter
from collections import defaultdict
import random
import os
from pathlib import Path
from itertools import combinations
import torch.nn.functional as F

def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
	
	fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
	fnr = 1 - tpr
	tunedThreshold = []
	if target_fr:
		for tfr in target_fr:
			idx = np.nanargmin(np.absolute((tfr - fnr)))
			tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
	for tfa in target_fa:
		idx = np.nanargmin(np.absolute((tfa - fpr))) # np.where(fpr<=tfa)[0][-1]
		tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
	idxE = np.nanargmin(np.absolute((fnr - fpr)))
	eer  = max(fpr[idxE],fnr[idxE])*100
	
	return tunedThreshold, eer, fpr, fnr

# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):

      # Sort the scores from smallest to largest, and also get the corresponding
      # indexes of the sorted scores.  We will treat the sorted scores as the
      # thresholds at which the the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      sorted_labels = []
      labels = [labels[i] for i in sorted_indexes]
      fnrs = []
      fprs = []

      # At the end of this loop, fnrs[i] is the number of errors made by
      # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
      # is the total number of times that we have correctly accepted scores
      # greater than thresholds[i].
      for i in range(0, len(labels)):
          if i == 0:
              fnrs.append(labels[i])
              fprs.append(1 - labels[i])
          else:
              fnrs.append(fnrs[i-1] + labels[i])
              fprs.append(fprs[i-1] + 1 - labels[i])
      fnrs_norm = sum(labels)
      fprs_norm = len(labels) - fnrs_norm

      # Now divide by the total number of false negative errors to
      # obtain the false positive rates across all thresholds
      fnrs = [x / float(fnrs_norm) for x in fnrs]

      # Divide by the total number of corret positives to get the
      # true positive rate.  Subtract these quantities from 1 to
      # get the false positive rates.
      fprs = [1 - x / float(fprs_norm) for x in fprs]
      return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold

def extract_id(path):
    p = Path(path)
    parts = p.parts
    idx = parts.index("voxceleb2")
    return parts[idx + 1]   # speaker id (e.g. id08533)


def relative_to_voxceleb2(path):
    p = Path(path)
    parts = p.parts
    idx = parts.index("voxceleb2")
    return Path(*parts[idx + 1:])
    
def sample_with_pairs(files, n, seed=42):
    rng = random.Random(seed)

    # 1. Sort input files (CRITICAL)
    files = sorted(files)

    # 2. Group files by speaker ID (deterministic)
    id_to_files = defaultdict(list)
    for f in files:
        id_to_files[extract_id(f)].append(f)

    # 3. Keep only IDs with at least 2 files (sorted)
    paired_ids = {
        k: sorted(v)
        for k, v in id_to_files.items()
        if len(v) >= 2
    }

    paired_speakers = sorted(paired_ids.keys())
    if not paired_speakers:
        raise ValueError("No speakers with at least 2 files found")

    target_paired = n // 2
    selected = []

    # 4. Select files in true pairs
    while len(selected) < target_paired:
        spk = rng.choice(paired_speakers)

        available_files = [
            f for f in paired_ids[spk] if f not in selected
        ]

        if len(available_files) < 2:
            continue

        pair = rng.sample(available_files, 2)
        selected.extend(pair)

    # 5. Fill remaining slots deterministically
    remaining = n - len(selected)
    remaining_pool = [
        f for f in files if f not in selected
    ]

    selected.extend(rng.sample(remaining_pool, remaining))

    return selected

def generate_pairs(subset, seed=42, output_file='/home/S3G-LABS/u1004/Documents/dataset/voxceleb/veri_train.txt', count = 100):
    """
    Given a subset of files, generate pairs:
    - 1 if same speaker
    - 0 if different speaker
    Writes result to `output_file`.
    """
    if os.path.exists(output_file):
        return

    rng = random.Random(seed)

    positive_pairs = []
    negative_pairs = []

    # Step 1: split pairs by label
    for f1, f2 in combinations(subset, 2):
        if extract_id(f1) == extract_id(f2):
            positive_pairs.append((f1, f2))
        else:
            negative_pairs.append((f1, f2))

    if len(positive_pairs) < count // 4:
        raise ValueError(
            f"Not enough positive pairs ({len(positive_pairs)}) "
            f"to satisfy 25% of {count}"
        )

    # Step 2: sample required positives and remaining negatives
    num_pos = count // 4
    num_neg = count - num_pos

    selected_pos = rng.sample(positive_pairs, num_pos)
    selected_neg = rng.sample(negative_pairs, num_neg)

    selected_pairs = selected_pos + selected_neg
    rng.shuffle(selected_pairs)

    # Step 3: write to file
    lines = []
    for f1, f2 in selected_pairs:
        label = 1 if extract_id(f1) == extract_id(f2) else 0
        f1_rel = relative_to_voxceleb2(f1)
        f2_rel = relative_to_voxceleb2(f2)
        lines.append(f"{label} {f1_rel} {f2_rel}")

    with open(output_file, "w") as f:
        f.write("\n".join(lines))

def fix_length_2d(tensor, target_len=74752):
    D = tensor.size(1)
    
    if D > target_len:
        # Truncate along dim=1
        tensor_fixed = tensor[:, :target_len]
    elif D < target_len:
        # Pad with zeros along dim=1
        pad_len = target_len - D
        # pad format for 2D: (left, right) for last dimension
        tensor_fixed = F.pad(tensor, (0, pad_len))
    else:
        tensor_fixed = tensor
        
    return tensor_fixed