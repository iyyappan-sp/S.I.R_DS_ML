from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# --- Configuration & Assets ---
# Load the pre-trained bundle (model, encoders, etc.)
try:
    bundle = joblib.load('scheme_project.pkl')
    model = bundle['model']
    encoders = bundle['encoders']
    edu_levels = bundle['edu_levels']
    coastal_districts = set(bundle['coastal'])
    urban_districts = set(bundle['urban'])

    # Load the scheme database for metadata and eligibility checks
    schemes_df = pd.read_csv('tn_schemes.csv')
    print("[INFO] Model and scheme data loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load assets: {e}")


# --- Helper Logic ---

def get_area_type(district):
    """Categorizes district into Coastal, Urban, or Rural."""
    if district in coastal_districts: return 'Coastal'
    if district in urban_districts: return 'Urban'
    return 'Rural'


def check_eligibility(person, scheme):
    """Business rules logic to filter out impossible schemes."""
    # 1. Category specific checks (Disability/Transgender)
    if scheme['category'] == 'Disability':
        if scheme['scheme_name'] == 'Transgender Welfare Scheme':
            if person.get('gender') != 'Transgender': return False
        elif person.get('disability_status', 'No') != 'Yes':
            return False

    # 2. Caste matching
    p_caste = person.get('caste', 'General')
    s_caste = scheme['caste']
    if s_caste != 'Any':
        if s_caste == 'BC/MBC' and p_caste not in ('BC', 'MBC'): return False
        if s_caste == 'SC/ST' and p_caste not in ('SC', 'ST'): return False
        if s_caste not in ('BC/MBC', 'SC/ST') and p_caste != s_caste: return False

    # 3. Education ranking
    p_edu_rank = edu_levels.get(person.get('education', 'No Formal Education'), 0)
    s_edu_min = edu_levels.get(scheme['education'], 0)
    if scheme['education'] != 'Any' and p_edu_rank < s_edu_min: return False

    # 4. Demographic & Income checks
    return (
            (scheme['gender'] == 'Any' or person['gender'] == scheme['gender']) and
            (scheme['min_age'] <= person['age'] <= scheme['max_age']) and
            (person['annual_income'] <= scheme['income_limit']) and
            (scheme['marital_status'] == 'Any' or person['marital_status'] == scheme['marital_status']) and
            (scheme['district'] == 'Any' or person['district_type'] == scheme['district'])
    )


# --- API Endpoints ---

@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "message": "TN Government Scheme Recommendation API is running."
    })


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Get input data from request
        data = request.get_json()

        # Basic preprocessing
        person = {
            'age': int(data.get('age', 30)),
            'gender': data.get('gender', 'Female'),
            'caste': data.get('caste', 'BC'),
            'occupation': data.get('occupation', 'Unorganised Worker'),
            'education': data.get('education', '10th Pass'),
            'annual_income': float(data.get('annual_income', 100000)),
            'marital_status': data.get('marital_status', 'Married'),
            'disability_status': data.get('disability_status', 'No'),
            'district': data.get('district', 'Chennai'),
            'family_size': int(data.get('family_size', 4))
        }

        # Derive features
        area = get_area_type(person['district'])
        person['district_type'] = area
        edu_rank = edu_levels.get(person['education'], 0)

        # Prepare Feature Matrix for ML Scoring
        def encode_val(col, val):
            le = encoders[col]
            return int(le.transform([val])[0]) if val in le.classes_ else 0

        matrix = []
        eligible_schemes = []

        for _, scheme in schemes_df.iterrows():
            if check_eligibility(person, scheme):
                # Calculate ML features for scoring
                row = [
                    person['age'], encode_val('gender', person['gender']),
                    encode_val('caste', person['caste']), encode_val('occupation', person['occupation']),
                    encode_val('education', person['education']), person['annual_income'],
                    encode_val('marital_status', person['marital_status']),
                    encode_val('disability_status', person['disability_status']),
                    encode_val('district_type', area), person['family_size'], edu_rank,
                    scheme['min_age'], scheme['max_age'], scheme['income_limit'], scheme['benefit_amount'],
                    1, 1,  # age_in_range and income_ok are True if check_eligibility passed
                    round(person['annual_income'] / scheme['income_limit'], 4),
                    round(person['age'] / scheme['max_age'], 4)
                ]
                matrix.append(row)
                eligible_schemes.append(scheme.to_dict())

        if not eligible_schemes:
            return jsonify({"count": 0, "recommendations": [], "message": "No eligible schemes found."})

        # Get scores from Random Forest
        scores = model.predict_proba(np.array(matrix, dtype=float))[:, 1]

        # Combine results
        results = []
        for i, scheme in enumerate(eligible_schemes):
            scheme['match_score'] = round(float(scores[i]) * 100, 2)
            results.append(scheme)

        # Sort by score (descending)
        results = sorted(results, key=lambda x: x['match_score'], reverse=True)

        return jsonify({
            "count": len(results),
            "recommendations": results[:10]  # Return top 10
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    # Set debug=True only for development
    app.run(host='0.0.0.0', port=5000, debug=False)