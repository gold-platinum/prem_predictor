import joblib
import numpy as np


def load_model():
    model = joblib.load("../models/prematch_model.pkl")
    encoders = joblib.load("../models/prematch_encoders.pkl")
    return model, encoders


def safe_encode(encoder, value):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        print(f"Warning: '{value}' not in training data, using default encoding")
        return 0


def predict_match(model, encoders, home_team, away_team, day):

    le_home = encoders['home']
    le_away = encoders['away']
    le_day = encoders['day']
    le_result = encoders['result']

    features = np.array([[
        safe_encode(le_home, home_team),  # home_team_encoded
        safe_encode(le_away, away_team),  # away_team_encoded
        safe_encode(le_day, day),  # day_encoded
        20,  # round (default mid-season)
        1 if day in ['Sat', 'Sun'] else 0,  # is_weekend
        0.45,  # home_win_rate (home advantage)
        0.35,  # away_win_rate
        0.60,  # home_home_win_rate (strong home factor)
        0.25,  # away_away_win_rate
        1.5,  # home_avg_goals
        1.2,  # away_avg_goals
        1.0,  # home_avg_conceded
        1.4,  # away_avg_conceded
        8.0,  # home_recent_form (out of 15)
        6.5,  # away_recent_form
        1.6,  # home_ppg (points per game)
        1.1,  # away_ppg
        0,  # games_played_diff
        0.2  # goal_diff_advantage
    ]])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    result_code = le_result.inverse_transform([prediction])[0]

    prob_mapping = {}
    for i, outcome in enumerate(le_result.classes_):
        if outcome == 'H':
            prob_mapping['Home Win'] = probabilities[i]
        elif outcome == 'D':
            prob_mapping['Draw'] = probabilities[i]
        elif outcome == 'A':
            prob_mapping['Away Win'] = probabilities[i]

    if result_code == 'H':
        prediction_text = f"{home_team} Win"
    elif result_code == 'A':
        prediction_text = f"{away_team} Win"
    else:
        prediction_text = "Draw"

    return {
        'prediction': prediction_text,
        'probabilities': prob_mapping,
        'confidence': max(probabilities)
    }


def main():

    model, encoders = load_model()
    if model is None:
        return

    print("Model loaded successfully!")
    print()

    print("Enter match details:")
    home_team = input("Home team: ").strip()
    away_team = input("Away team: ").strip()
    day = input("Day of week (Mon/Tue/Wed/Thu/Fri/Sat/Sun): ").strip()

    if not home_team or not away_team:
        print("Error: Please enter both team names")
        return

    if home_team == away_team:
        print("Error: Teams must be different")
        return

    valid_days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    if day not in valid_days:
        print(f"Error: Day must be one of: {', '.join(valid_days)}")
        return

    print("\nMaking prediction...")

    try:
        result = predict_match(model, encoders, home_team, away_team, day)

        # Output results
        print("\n" + "=" * 50)
        print("PREDICTION RESULTS")
        print("=" * 50)
        print(f"Match: {home_team} vs {away_team}")
        print(f"Day: {day}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print()
        print("Probability Breakdown:")
        for outcome, prob in result['probabilities'].items():
            print(f"  {outcome}: {prob:.1%}")

    except Exception as e:
        print(f"Error making prediction: {e}")


if __name__ == "__main__":
    main()