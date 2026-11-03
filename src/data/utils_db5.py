import numpy as np

def get_db5_unified_label(exercise: int, label: int) -> int:
    """
    Maps 3 distinct Ninapro DB5 exercises into a unified 53-class space.
    Rest is always 0.
    E1 (12 gestures): 1-12 -> 1-12
    E2 (17 gestures): 1-17 -> 13-29 (offset +12)
    E3 (23 gestures): 1-23 -> 30-52 (offset +29)
    """
    if label == 0:
        return 0
    if exercise == 1:
        if not (1 <= label <= 12):
            raise ValueError(f"Label {label} out of bounds for E1 (1..12)")
        return int(label)
    elif exercise == 2:
        if not (1 <= label <= 17):
            raise ValueError(f"Label {label} out of bounds for E2 (1..17)")
        return int(label + 12)
    elif exercise == 3:
        if not (1 <= label <= 23):
            raise ValueError(f"Label {label} out of bounds for E3 (1..23)")
        return int(label + 29)
    else:
        raise ValueError(f"Unknown exercise {exercise}")

def get_subject_splits_db5(subjects, train_size=6, val_size=2, test_size=2, seed=42):
    """
    Given a list of subjects, deterministically split them using a seed.
    """
    rng = np.random.RandomState(seed)
    shuffled_subjects = rng.permutation(sorted(subjects))
    train_subs = shuffled_subjects[:train_size].tolist()
    val_subs = shuffled_subjects[train_size:train_size+val_size].tolist()
    test_subs = shuffled_subjects[train_size+val_size:train_size+val_size+test_size].tolist()
    return train_subs, val_subs, test_subs

