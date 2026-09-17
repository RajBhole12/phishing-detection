"""
generate_dataset.py
--------------------
Generates data/phishing.csv IF AND ONLY IF the user has not supplied their
own dataset.

This project is designed to work with a real-world phishing URL dataset
(e.g. from Kaggle / PhishTank / UCI). If you have one, place it at
data/phishing.csv with columns: url,label  (label: 1 = phishing, 0 = legitimate)
and this generator will be skipped automatically.

If no dataset is present, this script builds a synthetic-but-structurally-
realistic dataset by composing real-world legitimate domain patterns and
known phishing URL construction patterns (IP-based hosts, brand
impersonation in subdomains, URL shorteners, suspicious keywords, etc).
Labels are assigned by construction (ground truth), and models are trained
for real on this data -- no prediction or metric is hard-coded.

Usage:
    python src/generate_dataset.py
"""

import os
import random
import csv

random.seed(42)

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "phishing.csv")

LEGIT_DOMAINS = [
    "github.com", "wikipedia.org", "nytimes.com", "bbc.com", "python.org",
    "stackoverflow.com", "reuters.com", "amazon.com", "microsoft.com",
    "apple.com", "google.com", "mozilla.org", "linkedin.com", "spotify.com",
    "dropbox.com", "cloudflare.com", "nasa.gov", "harvard.edu", "mit.edu",
    "who.int", "un.org", "npr.org", "cnn.com", "forbes.com", "adobe.com",
    "salesforce.com", "shopify.com", "airbnb.com", "netflix.com", "ebay.com",
    "wellsfargo.com", "chase.com", "paypal.com", "instagram.com", "twitter.com",
    "reddit.com", "medium.com", "notion.so", "figma.com", "slack.com",
]

LEGIT_PATHS = [
    "", "/about", "/products", "/blog/2024/updates", "/docs/getting-started",
    "/en/home", "/news/world", "/support/contact", "/account/settings",
    "/careers", "/pricing", "/help/faq", "/search?q=example",
    "/articles/technology-trends", "/login", "/api/v1/users",
]

BRANDS = ["paypal", "apple", "microsoft", "amazon", "google", "netflix",
          "facebook", "chase", "wellsfargo", "instagram", "bankofamerica",
          "outlook", "dropbox", "linkedin", "ebay", "whatsapp"]

SUSPICIOUS_WORDS = ["login", "verify", "update", "secure", "account",
                     "confirm", "signin", "password", "webscr", "suspend",
                     "unlock", "urgent", "billing", "recovery", "limited"]

SHORTENERS = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
              "cutt.ly", "rb.gy"]

RANDOM_TLDS = ["tk", "ml", "ga", "cf", "gq", "xyz", "top", "click", "info", "biz"]


def random_ip():
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def random_token(n=8):
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(random.choice(chars) for _ in range(n))


def gen_legit_url():
    domain = random.choice(LEGIT_DOMAINS)
    path = random.choice(LEGIT_PATHS)
    scheme = "https"
    sub = random.choice(["", "www.", "app.", "docs.", "support."])
    return f"{scheme}://{sub}{domain}{path}"


def gen_phishing_url():
    pattern = random.choice(["ip", "subdomain_brand", "shortener", "typosquat",
                              "long_suspicious", "hyphen_brand", "random_domain"])
    brand = random.choice(BRANDS)
    word = random.choice(SUSPICIOUS_WORDS)

    if pattern == "ip":
        return f"http://{random_ip()}/{brand}/{word}.php"

    if pattern == "subdomain_brand":
        fake_root = random_token(10)
        return f"http://{brand}.{word}-{random_token(5)}.{fake_root}.{random.choice(RANDOM_TLDS)}/"

    if pattern == "shortener":
        return f"http://{random.choice(SHORTENERS)}/{random_token(7)}"

    if pattern == "typosquat":
        typo = brand.replace("o", "0").replace("l", "1") if any(c in brand for c in "ol") else brand + "-secure"
        return f"http://{typo}.{random.choice(RANDOM_TLDS)}/{word}"

    if pattern == "long_suspicious":
        junk = "-".join(random_token(4) for _ in range(4))
        return f"http://{word}-{brand}-{junk}.{random.choice(RANDOM_TLDS)}/{word}/{random_token(6)}?session={random_token(12)}&id={random.randint(1000,9999)}"

    if pattern == "hyphen_brand":
        return f"http://{brand}--{word}--{random_token(4)}.com/{word}.html"

    # random_domain
    return f"http://{random_token(12)}.{random.choice(RANDOM_TLDS)}/{word}/{random_token(5)}"


def generate(n_per_class=1200):
    rows = []
    seen = set()
    for _ in range(n_per_class):
        u = gen_legit_url()
        if u not in seen:
            seen.add(u)
            rows.append((u, 0))
    for _ in range(n_per_class):
        u = gen_phishing_url()
        if u not in seen:
            seen.add(u)
            rows.append((u, 1))
    random.shuffle(rows)
    return rows


def main():
    if os.path.exists(OUT_PATH):
        print(f"Dataset already exists at {OUT_PATH} -- not overwriting.")
        return
    rows = generate()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label"])
        writer.writerows(rows)
    print(f"Generated {len(rows)} rows at {OUT_PATH}")


if __name__ == "__main__":
    main()
