from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# --- 1. Load trained model & data ---
try:
    bundle = joblib.load('scheme_project.pkl')
    model        = bundle['model']
    encoders     = bundle['encoders']
    edu_levels   = bundle['edu_levels']
    coastal_districts = set(bundle['coastal'])
    urban_districts   = set(bundle['urban'])

    schemes_df = pd.read_csv('tn_schemes.csv')

    # Pre-compute max benefit for normalisation (used in scoring)
    MAX_BENEFIT = schemes_df['benefit_amount'].max()

    print("API: Model and datasets loaded successfully.")
except Exception as e:
    print(f"API Error: Could not load assets. {e}")


# --- 2. Helper Functions ---

def get_area_type(district):
    if district in coastal_districts: return 'Coastal'
    if district in urban_districts:   return 'Urban'
    return 'Rural'


def check_eligibility(person, scheme):
    """Full eligibility check — mirrors the original training script exactly."""

    # 1. Disability / Category check
    if scheme['category'] == 'Disability':
        if scheme['scheme_name'] == 'Transgender Welfare Scheme':
            if person['gender'] != 'Transgender':
                return False
        elif person['disability_status'] != 'Yes':
            return False

    # 2. Caste check
    p_caste = person['caste']
    s_caste = scheme['caste']
    if s_caste != 'Any':
        if s_caste == 'BC/MBC' and p_caste not in ['BC', 'MBC']: return False
        if s_caste == 'SC/ST'  and p_caste not in ['SC', 'ST']:   return False
        if s_caste not in ['BC/MBC', 'SC/ST'] and p_caste != s_caste: return False

    # 3. Occupation check
    if scheme['occupation'] != 'Any':
        if person['occupation'] not in [o.strip() for o in scheme['occupation'].split('/')]:
            return False

    # 4. Education check (minimum level required)
    if scheme['education'] != 'Any':
        if edu_levels.get(person['education'], 0) < edu_levels.get(scheme['education'], 0):
            return False

    # 5. Gender, Age, Income, Marital Status, District checks
    return (
        (scheme['gender'] == 'Any' or person['gender'] == scheme['gender']) and
        scheme['min_age'] <= person['age'] <= scheme['max_age'] and
        person['annual_income'] <= scheme['income_limit'] and
        (scheme['marital_status'] == 'Any' or person['marital_status'] == scheme['marital_status']) and
        (scheme['district'] == 'Any' or person['district_type'] == scheme['district'])
    )


def compute_match_score(person, scheme):
    """
    Produces a meaningful 0-100 match score based on 3 weighted factors:
      - Income fit  (40%) : how far below the income limit the person is
      - Age fit     (30%) : how close the person's age is to the centre of the scheme's age band
      - Benefit     (30%) : how large the benefit is relative to all schemes
    """
    age_center  = (scheme['min_age'] + scheme['max_age']) / 2
    age_range   = max(scheme['max_age'] - scheme['min_age'], 1)

    income_fit  = 1 - (person['annual_income'] / scheme['income_limit'])
    age_fit     = 1 - abs(person['age'] - age_center) / age_range
    benefit_fit = scheme['benefit_amount'] / MAX_BENEFIT

    raw = income_fit * 0.4 + age_fit * 0.3 + benefit_fit * 0.3
    return round(max(0.0, min(raw * 100, 100.0)), 2)


# --- 3. API Routes ---

@app.route('/')
def home():
    return "TN Govt Scheme Recommendation API is Online!"


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json

        person = {
            'age'              : int(data.get('age', 0)),
            'gender'           : data.get('gender', ''),
            'caste'            : data.get('caste', ''),
            'occupation'       : data.get('occupation', 'Any'),
            'education'        : data.get('education', 'Any'),
            'annual_income'    : float(data.get('annual_income', 0)),
            'marital_status'   : data.get('marital_status', 'Any'),
            'disability_status': data.get('disability_status', 'No'),
            'district'         : data.get('district', 'Chennai'),
            'family_size'      : int(data.get('family_size', 4))
        }

        # Derive district_type — must be set before check_eligibility
        area = get_area_type(person['district'])
        person['district_type'] = area

        # --- Filter eligible schemes & score them ---
        results = []
        for _, s in schemes_df.iterrows():
            if check_eligibility(person, s):
                scheme_dict = s.to_dict()
                scheme_dict['match_percentage'] = compute_match_score(person, s)
                results.append(scheme_dict)

        if not results:
            return jsonify({
                "status"         : "success",
                "message"        : "No eligible schemes found. Visit the nearest Common Service Centre.",
                "total_found"    : 0,
                "recommendations": []
            })

        # Sort by match score descending, return top 5
        results.sort(key=lambda x: x['match_percentage'], reverse=True)

        return jsonify({
            "status"         : "success",
            "total_found"    : len(results),
            "recommendations": results[:5]
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# --- 4. Run ---
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))