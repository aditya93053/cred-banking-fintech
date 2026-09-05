from .governance import review
def run_autogen_review(draft, context):
    # Deterministic compatibility layer representing the required two-agent review:
    # Policy-Compliance-Reviewer -> Final-Editor, max_turns=2.
    return review(draft, context)
