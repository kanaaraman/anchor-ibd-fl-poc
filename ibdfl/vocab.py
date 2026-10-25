"""OMOP CDM concepts used by the PoC.

Source codes (ICD-10, ATC, LOINC) are the real codes. Standard concept_ids are
the ones we believe are correct in the OHDSI vocabulary, but verify every id in
Athena (https://athena.ohdsi.org) before running against a real CDM instance.
Concepts with ids >= 2,000,000,000 are local, project-defined concepts, which is
the OMOP convention for things the standard vocabularies do not cover.
"""

GENDER_MALE = 8507
GENDER_FEMALE = 8532

# condition_occurrence
CROHNS = {"concept_id": 201606, "source": "ICD10:K50"}
ULCERATIVE_COLITIS = {"concept_id": 81893, "source": "ICD10:K51"}

# drug_exposure: index biologics
BIOLOGICS = {
    "infliximab": {"concept_id": 937368, "source": "ATC:L04AB02", "anti_tnf": True},
    "adalimumab": {"concept_id": 1119119, "source": "ATC:L04AB04", "anti_tnf": True},
    "vedolizumab": {"concept_id": 45774639, "source": "ATC:L04AG05", "anti_tnf": False},
    "ustekinumab": {"concept_id": 40161532, "source": "ATC:L04AC05", "anti_tnf": False},
}
DRUG_ORDER = list(BIOLOGICS)
AZATHIOPRINE = {"concept_id": 19014878, "source": "ATC:L04AX01"}
PREDNISOLONE = {"concept_id": 1550557, "source": "ATC:H02AB06"}

# measurement
CRP = {"concept_id": 3020460, "source": "LOINC:1988-5", "unit": "mg/L"}
CALPROTECTIN = {"concept_id": 2000000010, "source": "LOINC:38445-3", "unit": "ug/g"}
ALBUMIN = {"concept_id": 3024561, "source": "LOINC:1751-7", "unit": "g/L"}
HEMOGLOBIN = {"concept_id": 3000963, "source": "LOINC:718-7", "unit": "g/L"}

# observation: local concepts
MONTREAL = {"concept_id": 2000000001, "source": "MONTREAL"}  # value_as_string L1/L2/L3 or E1/E2/E3
SMOKING_CURRENT = {"concept_id": 2000000002, "source": "SMOKING_CURRENT"}  # value_as_number 0/1
OUTCOME_SFR52 = {"concept_id": 2000000100, "source": "SFR_W52"}  # steroid-free remission at week 52, 0/1
