import re
from labels import label_is_pii

# Keywords representing numbers in STT
NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "oh",
    "double",
    "triple",
    "ten",
    "eleven",
    "twelve",
}


def count_digits_and_number_words(text):
    """Counts actual digits and spelled-out number words."""
    tokens = text.lower().split()
    count = 0
    for t in tokens:
        if t.isdigit() or t in NUMBER_WORDS:
            count += 1
    return count


def validate_span(text, label):
    """Returns False if the span text clearly doesn't match the label requirements."""
    # PHONE: Needs at least 7 number-like tokens (e.g. "nine eight four...")
    if label == "PHONE":
        if count_digits_and_number_words(text) < 7:
            return False

    # CREDIT_CARD: Needs at least 12 number-like tokens
    if label == "CREDIT_CARD":
        if count_digits_and_number_words(text) < 12:
            return False

    # EMAIL: Should usually contain "at" or "dot"
    if label == "EMAIL":
        if "at" not in text.lower() and "@" not in text:
            return False

    return True


def apply_conf_threshold(
    logits, ids, offsets, threshold=0.40
):  # Lowered threshold, we rely on validator
    import torch

    probs = torch.softmax(logits, dim=-1)
    conf = probs.max(dim=-1).values.tolist()

    new_ids = []
    for lid, c, (start, end) in zip(ids, conf, offsets):
        if start == 0 and end == 0:
            new_ids.append(lid)
            continue
        # We allow lower confidence because we will filter by structure later
        if c < threshold:
            new_ids.append(0)  # 'O'
        else:
            new_ids.append(lid)
    return new_ids
