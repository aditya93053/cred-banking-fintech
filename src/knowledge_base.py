KB = {
    "loan_eligibility_by_type": (
        "Personal Loan eligibility depends on age, income, credit history, "
        "and repayment capacity. Home, Auto, Education, and Business Loans "
        "can have additional type-specific eligibility requirements."
    ),

    "emi_rules": (
        "EMI depends on the principal amount, applicable interest rate, "
        "and loan tenure. A longer tenure can reduce the monthly EMI but "
        "may increase the total interest paid."
    ),

    "credit_card_fee_structure": (
        "Credit card fees can include annual fees, late-payment fees, "
        "cash-advance fees, and applicable transaction charges. "
        "The applicable fee depends on the card product and its terms."
    ),

    "kyc_documents": (
        "KYC generally requires identity and address verification documents. "
        "Accepted documents may include PAN, Aadhaar, passport, voter ID, "
        "or other officially accepted proof depending on the process."
    ),

    "fraud_dispute_process": (
        "For a suspected fraudulent transaction, the customer should report "
        "the transaction through the official banking support channel. "
        "The bank may verify the transaction and investigate the dispute."
    ),

    "account_closure": (
        "An account closure request should be submitted through the bank's "
        "approved process after clearing outstanding dues. "
        "The bank may require identity verification before closure."
    ),

    "interest_rate_slabs": (
        "Interest rates can vary according to loan type, borrower profile, "
        "credit history, amount, and applicable rate slab. "
        "The final applicable rate is determined according to the product terms."
    ),

    "prepayment_penalty": (
        "Prepayment charges may apply when a borrower repays a loan before "
        "the scheduled maturity date. The applicable charge depends on "
        "the loan product and its current terms."
    ),

    "minimum_balance": (
        "Some savings accounts may require a specified minimum balance. "
        "The required balance and applicable charges depend on the account "
        "type and the bank's terms."
    ),

    "credit_score_factors": (
        "Credit score can be affected by repayment history, credit utilization, "
        "credit history length, and the mix of credit accounts. "
        "Frequent credit applications can also affect credit profile."
    ),

    "joint_account_rules": (
        "Joint accounts can have multiple account holders with operating "
        "instructions defined during account opening. "
        "Transactions and closure requirements depend on the selected mandate."
    ),

    "nri_eligibility": (
        "NRI customers may be eligible for selected banking and loan products "
        "subject to applicable rules and documentation. "
        "Eligibility can depend on residency status, income, and product terms."
    ),
}


def get_document(topic):
    return KB.get(topic)


def all_documents():
    return KB.copy()


if __name__ == "__main__":
    print(f"Total KB documents: {len(KB)}")

    for topic, text in KB.items():
        print(f"\n{topic}")
        print(text)
