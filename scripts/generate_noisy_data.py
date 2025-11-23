import json
import random
import argparse
import os

# Expanded vocabulary for better generalization
PII_TYPES = ["EMAIL", "PHONE", "CREDIT_CARD", "PERSON_NAME", "DATE"]
NON_PII_TYPES = ["CITY", "LOCATION"]

# Mixed Indian and Western names to reflect realistic datasets
NAMES = [
    "john",
    "alice",
    "rohan",
    "sita",
    "michael",
    "anita",
    "rahul",
    "david",
    "priya",
    "vikram",
    "sarah",
    "emily",
    "mohammed",
    "arjun",
    "kavita",
    "steve",
    "rachel",
    "amit",
    "suresh",
    "deepa",
    "daniel",
    "jessica",
]

CITIES = [
    "mumbai",
    "delhi",
    "san francisco",
    "chennai",
    "bangalore",
    "new york",
    "london",
    "hyderabad",
    "pune",
    "chicago",
    "boston",
    "kolkata",
]

LOCATIONS = [
    "airport",
    "mall",
    "office",
    "school",
    "hospital",
    "station",
    "university",
    "bank",
    "park",
    "hotel",
    "cafe",
    "restaurant",
]

EMAIL_DOMAINS = [
    "gmail dot com",
    "yahoo dot com",
    "outlook dot com",
    "hotmail dot com",
    "company dot net",
    "school dot edu",
]

DIGIT_WORD = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

# STT Noise Patterns
PREFIXES = [
    "my",
    "the",
    "uh",
    "um",
    "here is",
    "it is",
    "contact",
    "call",
    "name is",
    "this is",
    "hello",
    "hi",
    "reach me at",
    "number is",
    "card number",
    "sent to",
    "living in",
    "going to",
]


def spell_number_noisy(num_str):
    """Simulates noisy STT number spoken patterns like 'double five'"""
    words = []
    i = 0
    while i < len(num_str):
        # 30% chance to say "double" for repeated digits
        if (
            i < len(num_str) - 1
            and num_str[i] == num_str[i + 1]
            and random.random() < 0.3
        ):
            words.append(f"double {DIGIT_WORD[num_str[i]]}")
            i += 2
        # 30% chance to say "triple"
        elif (
            i < len(num_str) - 2
            and num_str[i] == num_str[i + 1] == num_str[i + 2]
            and random.random() < 0.3
        ):
            words.append(f"triple {DIGIT_WORD[num_str[i]]}")
            i += 3
        else:
            words.append(DIGIT_WORD[num_str[i]])
            i += 1
    return " ".join(words)


def gen_email():
    name = random.choice(NAMES)
    domain = random.choice(EMAIL_DOMAINS)
    # Noisy variations: "at" vs " @"
    sep = random.choice([" at ", " @ "])
    return f"{name}{sep}{domain}"


def gen_phone():
    # 10 digit number
    num = "".join(random.choice("0123456789") for _ in range(10))
    return spell_number_noisy(num)


def gen_credit_card():
    # 16 digit number
    num = "".join(random.choice("0123456789") for _ in range(16))
    return spell_number_noisy(num)


def gen_date():
    d = random.randint(1, 28)
    m = random.randint(1, 12)
    y = random.randint(1990, 2025)
    # Randomly choose format: "january first", "1 1 2020", "first of jan"
    return f"{spell_number_noisy(str(d))} {spell_number_noisy(str(m))} {spell_number_noisy(str(y))}"


def gen_pii(label):
    if label == "EMAIL":
        return gen_email()
    if label == "PHONE":
        return gen_phone()
    if label == "CREDIT_CARD":
        return gen_credit_card()
    if label == "PERSON_NAME":
        return random.choice(NAMES)
    if label == "DATE":
        return gen_date()
    if label == "CITY":
        return random.choice(CITIES)
    if label == "LOCATION":
        return random.choice(LOCATIONS)
    return ""


def generate_example(example_id):
    # Mix PII and Non-PII
    labels = PII_TYPES + NON_PII_TYPES + ["O", "O"]  # Add dummy O for filler
    random.shuffle(labels)
    chosen = labels[: random.randint(1, 3)]

    text_parts = []
    entities = []
    current_len = 0

    for lab in chosen:
        if lab == "O":
            # Add filler text
            filler = random.choice(
                ["can you help", "please", "okay", "thanks", "hold on", "let me check"]
            )
            if current_len > 0:
                filler = " " + filler
            text_parts.append(filler)
            current_len += len(filler)
            continue

        ent_text = gen_pii(lab)
        prefix = random.choice(PREFIXES)

        # Construct chunk: "prefix entity"
        chunk = f"{prefix} {ent_text}" if current_len == 0 else f" {prefix} {ent_text}"

        start = (
            current_len + len(chunk) - len(ent_text)
        )  # calculate exact start of entity
        end = start + len(ent_text)

        text_parts.append(chunk)
        entities.append({"start": start, "end": end, "label": lab})
        current_len += len(chunk)

    full_text = "".join(text_parts).strip()
    return {"id": f"utt_{example_id}", "text": full_text, "entities": entities}


def main():
    ap = argparse.ArgumentParser()
    # Increased data size for better training
    ap.add_argument("--train", type=int, default=1500)
    ap.add_argument("--dev", type=int, default=300)
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        for i in range(args.train):
            f.write(json.dumps(generate_example(i)) + "\n")

    with open(os.path.join(args.out_dir, "dev.jsonl"), "w", encoding="utf-8") as f:
        for i in range(args.dev):
            f.write(json.dumps(generate_example(100000 + i)) + "\n")

    print(f"Generated {args.train} train, {args.dev} dev samples in {args.out_dir}")


if __name__ == "__main__":
    main()
