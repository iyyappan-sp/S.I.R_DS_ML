from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# --- 1. Load trained model & data ---
try:
    # This bundle contains the model, encoders, and district lists
    bundle = joblib.load('scheme_project.pkl')
    model = bundle['model']
    encoders = bundle['encoders']
    edu_levels = bundle['edu_levels']
    coastal_districts = set(bundle['coastal'])
    urban_districts = set(bundle['urban'])

    # Load scheme details for metadata
    schemes_df = pd.read_csv('tn_schemes.csv')
    print("API: Model and datasets loaded successfully.")
except Exception as e:
    print(f"API Error: Could not load assets. {e}")


# --- 2. Helper Functions ---

def get_area_type(district):
    if district in coastal_districts: return 'Coastal'
    if district in urban_districts: return 'Urban'
    return 'Rural'


def check_eligibility(person, scheme):
    """Checks basic rules before letting the ML model score it."""
    # Gender & Age
    if scheme['gender'] != 'Any' and person['gender'] != scheme['gender']: return False
    if not (scheme['min_age'] <= person['age'] <= scheme['max_age']): return False

    # Income & Caste
    if person['annual_income'] > scheme['income_limit']: return False

    p_caste = person['caste']
    s_caste = scheme['caste']
    if s_caste != 'Any':
        if s_caste == 'BC/MBC' and p_caste not in ['BC', 'MBC']: return False
        if s_caste == 'SC/ST' and p_caste not in ['SC', 'ST']: return False
        if s_caste not in ['BC/MBC', 'SC/ST'] and p_caste != s_caste: return False

    return True


# --- 3. API Routes ---

@app.route('/')
def home():
    return "TN Govt Scheme Recommendation API is Online!"


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Get data from user request
        data = request.json

        # Prepare person dictionary
        person = {
            'age': int(data.get('age', 0)),
            'gender': data.get('gender', ''),
            'caste': data.get('caste', ''),
            'occupation': data.get('occupation', 'Any'),
            'education': data.get('education', 'Any'),
            'annual_income': float(data.get('annual_income', 0)),
            'marital_status': data.get('marital_status', 'Any'),
            'disability_status': data.get('disability_status', 'No'),
            'district': data.get('district', 'Chennai'),
            'family_size': int(data.get('family_size', 4))
        }

        # Derive features
        area = get_area_type(person['district'])
        edu_rank = edu_levels.get(person['education'], 0)

        # Filter schemes and prepare scoring matrix
        valid_schemes = []
        matrix = []

        def encode(col, val):
            return int(encoders[col].transform([val])[0]) if val in encoders[col].classes_ else 0

        for _, s in schemes_df.iterrows():
            if check_eligibility(person, s):
                valid_schemes.append(s.to_dict())
                # Create the same 19 features used during training
                matrix.append([
                    person['age'], encode('gender', person['gender']), encode('caste', person['caste']),
                    encode('occupation', person['occupation']), encode('education', person['education']),
                    person['annual_income'], encode('marital_status', person['marital_status']),
                    encode('disability_status', person['disability_status']), encode('district_type', area),
                    person['family_size'], edu_rank,
                    s['min_age'], s['max_age'], s['income_limit'], s['benefit_amount'],
                    1, 1,  # age_in_range, income_ok
                    round(person['annual_income'] / s['income_limit'], 4),
                    round(person['age'] / s['max_age'], 4)
                ])

        if not valid_schemes:
            return jsonify({"message": "No eligible schemes found.", "results": []})

        # Predict match probability
        probs = model.predict_proba(np.array(matrix))[:, 1]

        # Attach scores and sort
        for i, scheme in enumerate(valid_schemes):
            scheme['match_percentage'] = round(float(probs[i] * 100), 2)

        sorted_results = sorted(valid_schemes, key=lambda x: x['match_percentage'], reverse=True)

        return jsonify({
            "status": "success",
            "total_found": len(sorted_results),
            "recommendations": sorted_results[:5]  # Return top 5
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)