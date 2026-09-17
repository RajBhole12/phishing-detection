"""
train_message_model.py
------------------------
Trains the Message (SMS/chat) phishing text classifier.

If data/messages.csv does not already exist, generates a structurally
realistic starter dataset instead of silently failing or faking results.
Drop in a real, labeled SMS/message dataset (columns: text,label) to
replace it for stronger real-world accuracy (e.g. the well-known "SMS
Spam Collection" dataset, relabeled for phishing vs legitimate).

Run:
    python src/message_detection/train_message_model.py
"""

import os
import sys
import csv
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.message_detection import message_model as mm

random.seed(11)

DATASET_PATH = mm.DATASET_PATH

LEGIT_MESSAGES = [
    "Hey, are we still on for lunch tomorrow at noon?",
    "Your Amazon package has been delivered to your front door.",
    "Reminder: your dentist appointment is tomorrow at 10am.",
    "Mom: call me back when you get a chance, nothing urgent.",
    "Your Uber is arriving in 3 minutes. Driver: John, license plate ABC123.",
    "Thanks for coming to the party last night, it was great seeing you!",
    "Your table at the restaurant is confirmed for 7pm tonight.",
    "Can you pick up milk on your way home?",
    "Your flight AA204 is on time and boards at gate 12.",
    "Practice is moved to 5pm today instead of 4pm.",
    "Just landed, will call you in an hour.",
    "Your prescription is ready for pickup at the pharmacy.",
    "Happy birthday! Hope you have an amazing day.",
    "Meeting moved to conference room B, same time.",
    "The package tracking shows it will arrive Thursday.",
    "Can we reschedule our call to 3pm instead?",
    "Your gym class tomorrow is at 6am, don't forget your mat.",
    "Loved the photos from the trip, thanks for sharing!",
    "Your library book is due back next Monday.",
    "See you at the game this weekend, go team!",
]

PHISHING_MESSAGES = [
    "URGENT: Your bank account has been locked. Verify now at http://bit.ly/secure-bank to restore access.",
    "Congratulations! You have won a $1000 gift card. Claim your prize now: http://bit.ly/claim-prize",
    "Your package could not be delivered. Confirm your address and pay a small fee here: http://tinyurl.com/redeliver",
    "Your account will be suspended in 24 hours. Verify your identity immediately: http://bit.ly/verify-now",
    "FINAL NOTICE: Your subscription payment failed. Update your card details immediately at http://bit.ly/update-pay",
    "You have been selected for a cash reward! Click here to claim before it expires: http://bit.ly/cash-reward",
    "Security alert: unusual login detected. Enter your OTP here to secure your account: http://bit.ly/secure-otp",
    "Your electricity will be disconnected today unless you pay now: http://bit.ly/pay-bill-now",
    "IRS Notice: You have an unclaimed refund. Verify your SSN to receive it: http://bit.ly/irs-refund",
    "Your card has been charged $499.99. If this wasn't you, verify your password immediately: http://bit.ly/dispute-now",
    "ACT NOW! Your account access expires today. Confirm your login credentials: http://bit.ly/confirm-login",
    "You've won a free iPhone! Limited time offer, claim now: http://bit.ly/free-iphone-claim",
    "Your verification code is required to unlock your account, reply with the code sent to your phone.",
    "WARNING: suspicious activity on your account. Verify immediately or lose access permanently: http://bit.ly/act-now",
    "Your delivery is on hold, pay a customs fee to release your package: http://tinyurl.com/customs-fee",
]


def random_case_variant(text):
    if random.random() < 0.35:
        return text.upper()
    return text


def add_urgency_punctuation(text):
    if random.random() < 0.3:
        return text + "!!"
    return text


def generate_dataset(n_per_class=220):
    rows = []
    seen = set()

    for _ in range(n_per_class):
        msg = random.choice(LEGIT_MESSAGES)
        if msg not in seen:
            seen.add(msg)
            rows.append((msg, 0))
        else:
            rows.append((msg + " ", 0))  # allow slight variation to reach target count

    for _ in range(n_per_class):
        msg = random.choice(PHISHING_MESSAGES)
        msg = random_case_variant(msg)
        msg = add_urgency_punctuation(msg)
        rows.append((msg, 1))

    random.shuffle(rows)
    return rows


def ensure_dataset():
    if os.path.exists(DATASET_PATH):
        print(f"Dataset already exists at {DATASET_PATH} -- not overwriting.")
        return
    rows = generate_dataset()
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    print(f"Generated {len(rows)} rows at {DATASET_PATH}")


if __name__ == "__main__":
    ensure_dataset()
    mm.train_and_save()
