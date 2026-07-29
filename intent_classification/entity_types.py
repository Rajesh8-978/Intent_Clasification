from __future__ import annotations

from enum import Enum


class EntityType(str, Enum):
    PROSPECT = "Prospect"
    CASE = "Case"
    CASE_BUSINESS = "CaseBusiness"
    CASE_TRUSTEE = "CaseTrustee"
    FILE = "File"
    USER = "User"
    TEMPLATE = "Template"
    CHAT_THREAD = "ChatThread"
    EMAIL_LOG = "EmailLog"
    PAYMENT_REQUEST = "PaymentRequest"
    CHEQUE_DEPOSIT_REQUEST = "ChequeDepositRequest"
    MISC_REQUEST = "MiscRequest"
    TRAVEL_APPLICATION = "TravelApplication"
    ICA_WATCH_LIST = "ICAWatchList"
    MCTC_PLAN = "MCTCPlan"
    STATEMENT_OF_AFFAIRS = "StatementOfAffairs"
    ASSET = "Asset"
    ASSET_CASH = "AssetCash"
    INSURANCE_POLICY = "InsurancePolicy"
    VEHICLE = "Vehicle"
    COMPANY_SHARES = "CompanyShares"
    REAL_ESTATE = "RealEstate"
    OTHER_ASSETS = "OtherAssets"
    MASS_LETTER_BATCH = "MassLetterBatch"
    MASS_LETTER_BATCH_RESPONSE_LOG = "MassLetterBatchResponseLog"
    MASS_LETTER_BATCH_DOC = "MassLetterBatchDoc"
    FIRST_LETTER_BATCH = "FirstLetterBatch"
    FIRST_LETTER_BATCH_DOC = "FirstLetterBatchDoc"
    SECOND_LETTER_BATCH_DOC = "SecondLetterBatchDoc"
    MASS_LETTER_BATCH_RESPONSE = "MassLetterBatchResponse"
    BUSINESS_ENTITY = "BusinessEntity"
    MASS_LETTER_BATCH_LETTER = "MassLetterBatchLetter"
    EMAIL = "Email"
    CASE_CREDITOR = "CaseCreditor"


LABEL_TO_ENTITY_TYPE: dict[str, EntityType] = {
    "Statement of Affairs": EntityType.STATEMENT_OF_AFFAIRS,
    "Payment Request": EntityType.PAYMENT_REQUEST,
    "Cheque Deposit Request": EntityType.CHEQUE_DEPOSIT_REQUEST,
    "Travel Application": EntityType.TRAVEL_APPLICATION,
    "Insurance Policy": EntityType.INSURANCE_POLICY,
    "Vehicle Documents": EntityType.VEHICLE,
    "Real Estate": EntityType.REAL_ESTATE,
    "Company Shares": EntityType.COMPANY_SHARES,
    "Case Trustee": EntityType.CASE_TRUSTEE,
    "Case Business": EntityType.CASE_BUSINESS,
    "Business Entity": EntityType.BUSINESS_ENTITY,
    "Case Creditor": EntityType.CASE_CREDITOR,
    "Asset": EntityType.ASSET,
    "Cash Asset": EntityType.ASSET_CASH,
    "Prospect": EntityType.PROSPECT,
    "General Case": EntityType.CASE,
    "Email": EntityType.EMAIL,
    "Miscellaneous Request": EntityType.MISC_REQUEST,
}


DEFAULT_INTENT_LABELS: tuple[str, ...] = tuple(LABEL_TO_ENTITY_TYPE.keys())


def map_label_to_entity_type(label: str) -> EntityType:
    try:
        return LABEL_TO_ENTITY_TYPE[label]
    except KeyError as exc:
        raise ValueError(f"Intent label is not mapped to an EntityType: {label!r}") from exc
