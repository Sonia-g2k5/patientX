"""
generate_dataset.py
--------------------
Creates a SYNTHETIC dataset for PatientX's price-prediction model.
No real patient data is used anywhere - every row is generated from
rules + randomness, which is exactly what you want for a placement demo
(explainable, reproducible, and zero privacy risk).

The price for each record is built from the criteria we're using to judge
data quality/value:
  - Hospital tier & accreditation      -> Hospital Credibility Score (HCS)
  - Disease category (rarity)          -> rarer disease = higher value
  - Treatment intensity (days/tests/surgery)
  - Data completeness (%)
  - Whether outcome data is included
  - Consent scope (narrow vs broad)
"""

import pandas as pd
import numpy as np

np.random.seed(42)

# ---------------------------------------------------------------
# 1. Reference tables (these ARE your "criteria")
# ---------------------------------------------------------------

# Hospital -> (tier 1-4, accredited 0/1)
# Tier: 1=Primary, 2=Secondary, 3=Tertiary, 4=Super-specialty
HOSPITALS = {
    'Apollo':              (4, 1),
    'CMC Vellore':         (4, 1),
    'MIOT':                (4, 1),
    'Global Hospital':     (4, 1),
    'Kauvery':             (3, 1),
    'SRM Hospital':        (3, 1),
    'Ramachandra':         (3, 1),
    'Meenakshi Mission':   (3, 1),
    'PSG':                 (3, 1),
    'Velammal':            (3, 0),
    'Vijaya Hospital':     (2, 1),
    'KG Hospital':         (2, 0),
    'Billroth':            (2, 0),
    'Stanley':             (2, 0),
    'Aarthi Scan':         (1, 0),
    'GH Chennai':          (2, 0),
    'GH Madurai':          (1, 0),
    'GH Trichy':           (1, 0),
    'GH Coimbatore':       (1, 0),
    'GH Salem':            (1, 0),
}

# Disease -> (category, rarity_weight)
DISEASES = {
    'Diabetes':          ('Common', 1.0),
    'Hypertension':       ('Common', 1.0),
    'Viral Fever':        ('Common', 0.8),
    'Asthma':             ('Chronic', 1.3),
    'Thyroid Disorder':   ('Chronic', 1.3),
    'Chronic Kidney Disease': ('Chronic', 1.6),
    'Cardiac Arrhythmia': ('Critical', 2.0),
    'Cancer':             ('Critical', 2.5),
    'Stroke':             ('Critical', 2.3),
    'Rare Genetic Disorder': ('Rare', 3.0),
    'Autoimmune Disorder': ('Rare', 2.6),
}

CONSENT_SCOPES = ['Single Study', 'Category of Research', 'Any Research Use']
CONSENT_WEIGHT = {'Single Study': 1.0, 'Category of Research': 1.3, 'Any Research Use': 1.6}


def hospital_credibility_score(tier, accredited):
    """HCS out of 100, combining tier + accreditation."""
    return tier * 20 + accredited * 20


def make_row():
    hosp = np.random.choice(list(HOSPITALS.keys()))
    tier, accredited = HOSPITALS[hosp]
    hcs = hospital_credibility_score(tier, accredited)

    disease = np.random.choice(list(DISEASES.keys()))
    category, rarity_weight = DISEASES[disease]

    days = np.random.randint(1, 10)
    tests = np.random.randint(1, 9)
    surgery = np.random.choice([0, 1], p=[0.7, 0.3])
    completeness = np.random.randint(50, 101)          # % of fields filled
    outcome_included = np.random.choice([0, 1], p=[0.4, 0.6])
    consent = np.random.choice(CONSENT_SCOPES)

    # ---- Price formula (this is the "ground truth" your model learns) ----
    base = 300
    price = (
        base
        + hcs * 4
        + rarity_weight * 350
        + days * 90
        + tests * 120
        + surgery * 1500
        + completeness * 8
        + outcome_included * 400
        + CONSENT_WEIGHT[consent] * 300
    )
    price += np.random.normal(0, 150)  # noise so it's not a perfect formula
    price = max(200, round(price, -1))  # floor + round to nearest 10

    return {
        'Hospital': hosp,
        'Hospital_Tier': tier,
        'Accreditation': accredited,
        'HCS': hcs,
        'Disease': disease,
        'Disease_Category': category,
        'Days_Admitted': days,
        'Tests_Count': tests,
        'Surgery': surgery,
        'Data_Completeness': completeness,
        'Outcome_Included': outcome_included,
        'Consent_Scope': consent,
        'Price': price,
    }


def main(n_rows=800, out_path='patientx_dataset.csv'):
    rows = [make_row() for _ in range(n_rows)]
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows -> {out_path}")
    print(df.head())


if __name__ == '__main__':
    main()
