"""Business descriptions used as hypotheses by the zero-shot NLI model."""

from __future__ import annotations


LABEL_DESCRIPTIONS: dict[str, str] = {
    "Statement of Affairs": (
        "a statement of affairs, insolvency financial disclosure, or a declaration "
        "of assets, liabilities, income, and expenses"
    ),
    "Payment Request": (
        "a request to make, approve, process, reimburse, or confirm a payment or invoice"
    ),
    "Cheque Deposit Request": (
        "depositing, banking, clearing, or recording a cheque"
    ),
    "Travel Application": (
        "a travel application, travel approval, itinerary, flight, accommodation, or travel expense"
    ),
    "Insurance Policy": (
        "an insurance policy, insurance coverage, premium, renewal, claim, or insurer document"
    ),
    "Vehicle Documents": (
        "vehicle registration, COE renewal, vehicle ownership, vehicle licence, vehicle number, "
        "registration details, or records from a vehicle or land transport authority"
    ),
    "Real Estate": (
        "land, a house, an apartment, commercial property, property ownership, title, mortgage, "
        "sale, valuation, or other real estate"
    ),
    "Company Shares": (
        "shares, stocks, securities, share certificates, dividends, or ownership of a company"
    ),
    "Case Trustee": (
        "the appointment, authority, identity, instructions, or actions of a trustee in a case"
    ),
    "Case Business": (
        "business operations, trading activity, business records, or business matters belonging "
        "to an existing case"
    ),
    "Business Entity": (
        "the identity, registration, ownership, structure, directors, or records of a company "
        "or other business entity"
    ),
    "Case Creditor": (
        "a creditor, proof of debt, creditor claim, amount owed, debt verification, or creditor "
        "details in an existing case"
    ),
    "Asset": (
        "a non-cash asset, property, possession, valuation, ownership, disposal, or recovery of "
        "an asset that has no more specific category"
    ),
    "Cash Asset": (
        "cash, a bank account, account balance, deposit, savings, or another liquid financial asset"
    ),
    "Prospect": (
        "a prospective client, new customer enquiry, lead, onboarding opportunity, or request "
        "to begin a new engagement"
    ),
    "General Case": (
        "general case administration, status, correspondence, or case information that has no "
        "more specific case category"
    ),
    "Email": (
        "email delivery, forwarding, mailbox handling, an email record, or an attachment issue "
        "where the message has no more specific business subject"
    ),
    "Miscellaneous Request": (
        "a request that genuinely does not match any of the other available business categories"
    ),
}


def build_label_hypothesis(label: str) -> str:
    """Expand a short business label into a model-readable hypothesis."""

    description = LABEL_DESCRIPTIONS.get(label)
    # Custom runtime labels remain supported even when no detailed definition exists.
    if description is None:
        return f"The primary business intent of this message is {label}."
    return f"The primary business intent of this message is {label}: {description}."
