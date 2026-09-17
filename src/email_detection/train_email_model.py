"""
train_email_model.py
----------------------
Trains the Email phishing text classifier.

If data/emails.csv does not already exist, generates a structurally
realistic starter dataset (same transparent approach used by
src/generate_dataset.py for the URL detector) instead of silently failing
or faking results. Drop in a real, labeled email dataset (columns:
text,label) to replace it for stronger real-world accuracy.

Run:
    python src/email_detection/train_email_model.py
"""

import os
import sys
import csv
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.email_detection import email_model as em

random.seed(7)

DATASET_PATH = em.DATASET_PATH

LEGIT_SENDERS = [
    "notifications@github.com", "no-reply@dropbox.com", "team@slack.com",
    "updates@linkedin.com", "receipts@amazon.com", "calendar@google.com",
    "newsletter@nytimes.com", "hr@mycompany.com", "billing@spotify.com",
    "support@notion.so", "no-reply@zoom.us", "orders@shopify.com",
]

LEGIT_SUBJECTS_BODIES = [
    ("Your weekly project summary", "Here's a quick recap of what changed in your project this week. Three pull requests were merged and two issues were closed. Have a great week!"),
    ("Your order has shipped", "Good news! Your recent order has shipped and is on its way. You can track the package using the carrier's tracking number in your account order history."),
    ("Meeting reminder: Team sync at 3pm", "Just a reminder that our team sync is scheduled for 3pm today in the usual conference room. Agenda is attached. See you there."),
    ("Your monthly statement is ready", "Your monthly account statement is now available to view in your online dashboard. No action is needed unless you have questions about a transaction."),
    ("Welcome to the newsletter", "Thanks for subscribing! Every week we'll send a short roundup of the most interesting articles from around the web. You can unsubscribe anytime from the footer."),
    ("Your subscription receipt", "This confirms your subscription renewal for this month. Your next billing date is noted in your account settings. Thank you for being a subscriber."),
    ("New comment on your document", "Someone left a comment on the document you shared. Open the document to view the comment and reply if needed."),
    ("Reminder: your appointment tomorrow", "This is a friendly reminder about your appointment scheduled for tomorrow. Please arrive a few minutes early. Reply to this email if you need to reschedule."),
    ("Your flight itinerary", "Attached is your confirmed flight itinerary for the upcoming trip. Please review the details and let us know if anything needs to change."),
    ("Product update: new features this month", "We shipped several new features this month based on your feedback. Read the full release notes on our blog for details."),
]

PHISHING_SENDERS = [
    "security@paypa1-support.com", "no-reply@account-verify-secure.tk",
    "support@bankalert-update.info", "admin@login-verification.xyz",
    "billing@refund-department.click", "team@urgent-notice.top",
    "helpdesk@it-supportdesk.ml", "alerts@amaz0n-security.com",
]

PHISHING_TEMPLATES = [
    ("URGENT: Your account will be suspended",
     "We detected unusual activity on your account. Your account will be suspended within 24 hours unless you verify your identity immediately. Click here to confirm your password and avoid permanent suspension: http://secure-verify-login.tk/confirm"),
    ("Action required: Verify your payment information",
     "Your last payment could not be processed. To avoid service interruption, please update your billing information and confirm your credit card number within 24 hours. http://billing-update-now.xyz/pay"),
    ("Congratulations! You have won a reward",
     "Congratulations! You have been selected to receive a special reward. Claim your prize now by verifying your account details and providing your bank account number before the offer expires today."),
    ("Security Alert: unusual sign-in attempt",
     "We noticed an unusual sign-in attempt on your account from a new device. If this wasn't you, verify your identity immediately by entering your one-time password and current login credentials here: http://account-verify-secure.tk"),
    ("Your invoice is overdue - immediate action required",
     "Your invoice is now overdue. Please make a wire transfer or send payment via gift card immediately to avoid additional late fees and legal action. Reply with your bank account number to confirm."),
    ("Final notice: reset your password now",
     "This is your final notice. For security reasons you must reset your password immediately by clicking the link below and entering your current password and security code: http://reset-password-now.click"),
    ("IT Support: mailbox quota exceeded, verify now",
     "Your mailbox has exceeded its storage quota. To avoid losing access, verify your account immediately by entering your username and password at the link below within 24 hours."),
    ("Refund department: claim your refund today",
     "You are eligible for a refund. To process your refund today, confirm your bank account number, date of birth, and social security number using the secure form linked below."),
]


def random_case_variant(text):
    if random.random() < 0.3:
        words = text.split()
        idx = random.randrange(len(words))
        words[idx] = words[idx].upper()
        return " ".join(words)
    return text


def generate_dataset(n_per_class=260):
    rows = []
    seen = set()

    for _ in range(n_per_class):
        sender = random.choice(LEGIT_SENDERS)
        subject, body = random.choice(LEGIT_SUBJECTS_BODIES)
        text = f"From: {sender}\nSubject: {subject}\n{body}"
        if text not in seen:
            seen.add(text)
            rows.append((text, 0))

    for _ in range(n_per_class):
        sender = random.choice(PHISHING_SENDERS)
        subject, body = random.choice(PHISHING_TEMPLATES)
        subject = random_case_variant(subject)
        extra_punct = "!" * random.choice([0, 0, 1, 2, 3])
        text = f"From: {sender}\nSubject: {subject}{extra_punct}\n{body}"
        if text not in seen:
            seen.add(text)
            rows.append((text, 1))

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
    em.train_and_save()
