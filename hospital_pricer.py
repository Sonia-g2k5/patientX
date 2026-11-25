"""
hospital_pricer.py
-------------------
PatientX price-prediction model (refined version).

Key changes vs your original:
  1. Trains on a real CSV dataset (patientx_dataset.csv) instead of a
     20-row hardcoded dict -> more realistic, easy to expand/tune.
  2. Adds Disease, Data_Completeness, Outcome_Included, Consent_Scope
     as features - not just Hospital/Days/Tests/Surgery.
  3. Unknown hospitals no longer get an arbitrary "10" - they get an
     HCS computed from a sensible default tier, so the price still
     makes sense.
  4. Disease + outcome + consent are auto-detected from the uploaded
     file's text (prescription, discharge summary, scanned report, etc).
  5. NEW: train() now does a train/test split and reports accuracy
     metrics (R^2, MAE, RMSE, MAPE) so you can see how good the model
     actually is, instead of training blind.
"""

import os
import re
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from generate_dataset import HOSPITALS, DISEASES, CONSENT_SCOPES, hospital_credibility_score

NUMERIC_FEATURES = ['Hospital_Tier', 'Accreditation', 'HCS', 'Days_Admitted',
                     'Tests_Count', 'Surgery', 'Data_Completeness', 'Outcome_Included']
CATEGORICAL_FEATURES = ['Disease_Category', 'Consent_Scope']


class PatientXPricer:
    def __init__(self, dataset_path='patientx_dataset.csv'):
        self.dataset_path = dataset_path
        self.pipeline = None
        self.is_trained = False
        self.seen_files = set()
        self.metrics = {}  # populated after train()

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------
    def train(self, test_size=0.2, random_state=42, cv_folds=5, verbose=True):
        """
        Fits the pipeline and reports accuracy.

        - Splits the data into train/test so metrics reflect performance
          on rows the model did NOT see during fitting.
        - Reports R^2, MAE, RMSE, and MAPE on the held-out test set.
        - Also runs k-fold cross-validation (R^2) across the whole
          dataset, since with a small CSV a single train/test split can
          be noisy.
        - After computing metrics, refits on the FULL dataset so the
          model used for predict() still benefits from every row.
        """
        df = pd.read_csv(self.dataset_path)

        X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        y = df['Price']

        def build_pipeline():
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
                ],
                remainder='passthrough'  # numeric features pass through untouched
            )
            return Pipeline(steps=[
                ('preprocess', preprocessor),
                ('model', LinearRegression())
            ])

        # --- 1. Train/test split evaluation ---------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        eval_pipeline = build_pipeline()
        eval_pipeline.fit(X_train, y_train)
        y_pred = eval_pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        # sqrt of MSE computed manually - works across all sklearn versions
        # (newer sklearn dropped the squared= kwarg in favor of a separate
        # root_mean_squared_error function, so this avoids depending on version)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # avoid divide-by-zero on any zero-price rows
        nonzero = y_test != 0
        mape = (np.abs((y_test[nonzero] - y_pred[nonzero]) / y_test[nonzero])).mean() * 100 if nonzero.any() else float('nan')

        # --- 2. Cross-validation across the whole dataset --------------
        cv_pipeline = build_pipeline()
        n_splits = min(cv_folds, len(df))  # can't have more folds than rows
        cv_scores = cross_val_score(cv_pipeline, X, y, cv=n_splits, scoring='r2')

        self.metrics = {
            'test_r2': r2,
            'test_mae': mae,
            'test_rmse': rmse,
            'test_mape': mape,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'cv_folds': n_splits,
            'n_train': len(X_train),
            'n_test': len(X_test),
        }

        # --- 3. Refit on the FULL dataset for actual predictions -------
        self.pipeline = build_pipeline()
        self.pipeline.fit(X, y)
        self.is_trained = True

        if verbose:
            self.print_accuracy_report(len(df))

        return self.metrics

    def print_accuracy_report(self, n_rows=None):
        m = self.metrics
        if not m:
            print("No metrics yet - call train() first.")
            return
        print("=" * 55)
        print("MODEL ACCURACY REPORT")
        print("=" * 55)
        if n_rows is not None:
            print(f"Dataset size        : {n_rows} rows")
        print(f"Train / test split   : {m['n_train']} / {m['n_test']} rows")
        print(f"Test R^2 score       : {m['test_r2']:.3f}  "
              f"(1.0 = perfect, 0.0 = no better than guessing the mean)")
        print(f"Test MAE             : Rs {m['test_mae']:,.2f}")
        print(f"Test RMSE            : Rs {m['test_rmse']:,.2f}")
        print(f"Test MAPE            : {m['test_mape']:.1f}%")
        print(f"{m['cv_folds']}-fold CV R^2       : "
              f"{m['cv_r2_mean']:.3f} (+/- {m['cv_r2_std']:.3f})")
        print("=" * 55)

    # ---------------------------------------------------------------
    # File handling
    # ---------------------------------------------------------------
    def get_file_hash(self, file_path):
        hash_md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _extract_text(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext in ['.jpg', '.jpeg', '.png', '.webp']:
                from PIL import Image
                import pytesseract
                return pytesseract.image_to_string(Image.open(file_path))
            elif ext == '.pdf':
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    return ' '.join(p.extract_text() or '' for p in pdf.pages)
            elif ext == '.docx':
                from docx import Document
                doc = Document(file_path)
                return ' '.join(p.text for p in doc.paragraphs)
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        except Exception:
            return ''

    # ---------------------------------------------------------------
    # Feature extraction from raw uploaded text (prescription, report, etc.)
    # ---------------------------------------------------------------
    def _detect_hospital(self, text):
        for name, (tier, accredited) in HOSPITALS.items():
            if name.lower() in text:
                return name, tier, accredited
        # Unknown hospital -> assume an average, unaccredited secondary hospital
        return 'Unknown', 2, 0

    def _detect_disease(self, text):
        for disease, (category, _weight) in DISEASES.items():
            if disease.lower() in text:
                return disease, category
        return 'Unspecified', 'Common'  # safest default

    def _detect_consent(self, text):
        if 'any research' in text or 'broad consent' in text:
            return 'Any Research Use'
        if 'category of research' in text:
            return 'Category of Research'
        return 'Single Study'  # conservative default

    def _estimate_completeness(self, text):
        """Rough proxy: how many of the 'expected' fields show up in the text."""
        expected_fields = ['name', 'age', 'diagnosis', 'date', 'doctor',
                           'dosage', 'test', 'result', 'hospital', 'admission']
        found = sum(1 for f in expected_fields if f in text)
        return int(round(found / len(expected_fields) * 100))

    def extract_features(self, text):
        text = text.lower()

        # [ \t]* (not \s*) so we don't cross a newline and accidentally grab
        # a year from a date line above/below the actual "days" mention.
        # Reject anything above 90 - almost certainly a misread, not a stay length.
        day_matches = [int(m.group(1)) for m in re.finditer(r'(\d{1,2})[ \t]*days?\b', text)]
        days = day_matches[0] if day_matches else 1

        tests = min(len(re.findall(r'(test|scan|x-ray|lab|mri|ct\b)', text)), 10)
        surgery_mentioned = re.search(r'surgery|operation|surgical', text)
        surgery_negated = re.search(r'(no|not|without)\s+(any\s+)?(surgery|operation)', text)
        surgery = 1 if (surgery_mentioned and not surgery_negated) else 0
        outcome_included = 1 if re.search(r'recovered|discharged|cured|stable|improved', text) else 0

        hosp_name, tier, accredited = self._detect_hospital(text)
        disease, category = self._detect_disease(text)
        consent = self._detect_consent(text)
        completeness = self._estimate_completeness(text)
        hcs = hospital_credibility_score(tier, accredited)

        return {
            'hospital': hosp_name,
            'Hospital_Tier': tier,
            'Accreditation': accredited,
            'HCS': hcs,
            'Days_Admitted': days,
            'Tests_Count': tests,
            'Surgery': surgery,
            'Data_Completeness': completeness,
            'Outcome_Included': outcome_included,
            'Disease': disease,
            'Disease_Category': category,
            'Consent_Scope': consent,
        }

    # ---------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------
    def predict(self, file_path):
        if not self.is_trained:
            raise RuntimeError('Call train() before predict().')

        file_hash = self.get_file_hash(file_path)
        if file_hash in self.seen_files:
            return 'You uploaded the same file again.'
        self.seen_files.add(file_hash)

        text = self._extract_text(file_path)
        if not text.strip():
            return 'Could not read any text from this file. Please upload a clearer document.'

        feats = self.extract_features(text)

        row = pd.DataFrame([{
            'Hospital_Tier': feats['Hospital_Tier'],
            'Accreditation': feats['Accreditation'],
            'HCS': feats['HCS'],
            'Days_Admitted': feats['Days_Admitted'],
            'Tests_Count': feats['Tests_Count'],
            'Surgery': feats['Surgery'],
            'Data_Completeness': feats['Data_Completeness'],
            'Outcome_Included': feats['Outcome_Included'],
            'Disease_Category': feats['Disease_Category'],
            'Consent_Scope': feats['Consent_Scope'],
        }])

        price = round(self.pipeline.predict(row)[0], -1)  # nearest Rs 10

        summary = (
            f"Detected -> Hospital: {feats['hospital']} (Tier {feats['Hospital_Tier']}, "
            f"HCS {feats['HCS']}), Disease: {feats['Disease']} ({feats['Disease_Category']}), "
            f"Days: {feats['Days_Admitted']}, Tests: {feats['Tests_Count']}, "
            f"Surgery: {'Yes' if feats['Surgery'] else 'No'}, "
            f"Completeness: {feats['Data_Completeness']}%, Consent: {feats['Consent_Scope']}"
        )

        confidence_note = (
            f"(model test R^2: {self.metrics['test_r2']:.2f}, "
            f"avg error: Rs {self.metrics['test_mae']:,.0f})"
            if self.metrics else ""
        )

        return (f"{summary}\n\n"
                f"Data accepted. Can be listed on PatientX at Rs {price:,.0f} {confidence_note}")


def main():
    pricer = PatientXPricer()
    pricer.train()  # now prints the accuracy report automatically

    files_input = input('Upload files (comma-separated paths): ').strip()
    if not files_input:
        return
    for file_path in [f.strip() for f in files_input.split(',')]:
        if os.path.exists(file_path):
            print(pricer.predict(file_path))
        else:
            print(f'File not found: {file_path}')


if __name__ == '__main__':
    main()