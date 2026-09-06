import re
PII_PATTERNS = [
    (re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"), "[PAN-MASKED]"),
    (re.compile(r"\b\d{12}\b"), "[AADHAAR-MASKED]"),
    (re.compile(r"\b\d{9,18}\b"), "[ACCOUNT-MASKED]"),
]
INJECTION = re.compile(r"(ignore\s+(all|previous)\s+instructions|reveal\s+system\s+prompt|jailbreak|developer\s+message)", re.I)

def mask_pii(text: str) -> str:
    for p, repl in PII_PATTERNS:
        text = p.sub(repl, text)
    return text

def is_prompt_injection(text: str) -> bool:
    return bool(INJECTION.search(text))

def in_scope(text: str) -> bool:
    terms = ["loan","emi","credit card","kyc","fraud","dispute","account closure","interest","prepayment","minimum balance","credit score","joint account","nri","application status"]
    return any(t in text.lower() for t in terms)
