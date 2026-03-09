"""Scoring system for shot mechanics analysis."""

from feedback import generate_coaching_feedback


# Weights for each mechanic (must sum to 100)
WEIGHTS = {
    'weight_transfer': 30,
    'hip_rotation': 15,
    'shoulder_rotation': 15,
    'shot_loading': 15,
    'hand_separation': 10,
    'timing_sequence': 15,
}


def compute_shot_score(mechanics: dict) -> dict:
    """Compute overall shot score from individual mechanics.

    Args:
        mechanics: Dict from analyze_mechanics()

    Returns:
        Dict with total_score, breakdown, and feedback.
    """
    breakdown = {}
    total_score = 0.0

    for mechanic, weight in WEIGHTS.items():
        if mechanic in mechanics:
            score = mechanics[mechanic].get('score', 0.0)
            points = score * weight
            breakdown[mechanic] = {
                'points': round(points, 1),
                'max_points': weight,
                'rating': mechanics[mechanic].get('rating', 'unknown'),
                'detail': mechanics[mechanic].get('detail', ''),
            }
            total_score += points
        else:
            breakdown[mechanic] = {
                'points': 0,
                'max_points': weight,
                'rating': 'unknown',
                'detail': 'Not analyzed',
            }

    total_score = round(total_score, 1)
    feedback = generate_coaching_feedback(total_score, mechanics)

    return {
        'total_score': total_score,
        'breakdown': breakdown,
        'feedback': feedback,
    }
