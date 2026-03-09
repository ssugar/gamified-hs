"""Youth-friendly coaching feedback for shot mechanics."""


COACHING_TIPS = {
    'weight_transfer': {
        'weak': "Try starting with more weight on your back foot and driving your hips forward during the shot.",
        'moderate': "You're shifting your weight a bit - push even more from your back foot to your front foot as you shoot.",
        'good': "Nice weight transfer! Keep driving forward through the shot.",
        'excellent': "Excellent weight transfer - your whole body is moving into the shot!",
    },
    'hip_rotation': {
        'weak': "Try turning your hips toward the net as you shoot - think about pointing your belly button at the target.",
        'moderate': "Your hips are starting to turn - really snap them toward the net for more power.",
        'good': "Good hip rotation! Your lower body is helping power the shot.",
        'excellent': "Great hip rotation - you're really driving with your lower body!",
    },
    'shoulder_rotation': {
        'weak': "Rotate your shoulders toward the target as you shoot - your chest should face the net at the end.",
        'moderate': "Your shoulders are turning some - try to get your chest pointing at the net by the end of the shot.",
        'good': "Nice shoulder rotation! You're using your upper body well.",
        'excellent': "Excellent shoulder rotation - great upper body mechanics!",
    },
    'shot_loading': {
        'weak': "Bring the puck further back behind your body before shooting - this loads the stick for more power.",
        'moderate': "You're loading the shot a bit - try pulling your hands back more before releasing.",
        'good': "Good shot loading - you're getting the puck back before shooting.",
        'excellent': "Great shot loading - you're really winding up for power!",
    },
    'hand_separation': {
        'weak': "Pull your top hand away from your body as you shoot to load the stick and get more flex.",
        'moderate': "Your hands are separating some - try to push your bottom hand forward while pulling the top hand back.",
        'good': "Good hand separation - you're getting some stick flex.",
        'excellent': "Excellent hand separation - you're really loading the stick!",
    },
    'timing_sequence': {
        'weak': "Start the shot with your hips and shoulders, not just your arms. Think: hips, then shoulders, then hands.",
        'moderate': "Your timing is getting there - focus on starting the rotation from your hips first.",
        'good': "Good sequence! Your body parts are firing in the right order.",
        'excellent': "Perfect timing - hips lead, shoulders follow, then the release. Textbook!",
    },
}


def generate_coaching_feedback(
    total_score: float,
    mechanics: dict
) -> list[str]:
    """Generate youth-friendly coaching feedback.

    Args:
        total_score: Overall shot score 0-100
        mechanics: Dict from analyze_mechanics()

    Returns:
        List of feedback strings
    """
    feedback = []

    # Overall assessment
    if total_score >= 80:
        feedback.append("Awesome shot! Your mechanics look really strong.")
    elif total_score >= 60:
        feedback.append("Good shot! A few tweaks will make it even better.")
    elif total_score >= 40:
        feedback.append("Nice effort! Let's work on a few things to level up your shot.")
    else:
        feedback.append("Keep practicing! Here's what to focus on to improve your shot.")

    # Collect weak areas sorted by weight importance
    from scoring import WEIGHTS
    weak_areas = []
    for mechanic in sorted(WEIGHTS, key=WEIGHTS.get, reverse=True):
        if mechanic in mechanics:
            rating = mechanics[mechanic].get('rating', 'unknown')
            if rating in ('weak', 'moderate'):
                weak_areas.append((mechanic, rating))

    # Show up to 3 coaching tips for weak areas
    if weak_areas:
        feedback.append("")
        feedback.append("Things to work on:")
        for mechanic, rating in weak_areas[:3]:
            tip = COACHING_TIPS.get(mechanic, {}).get(rating, '')
            if tip:
                name = mechanic.replace('_', ' ').title()
                feedback.append(f"  {name}: {tip}")

    # Highlight strengths
    strong_areas = []
    for mechanic in WEIGHTS:
        if mechanic in mechanics:
            rating = mechanics[mechanic].get('rating', 'unknown')
            if rating in ('excellent', 'good'):
                strong_areas.append(mechanic)

    if strong_areas:
        feedback.append("")
        names = [a.replace('_', ' ').title() for a in strong_areas]
        feedback.append(f"Strengths: {', '.join(names)}")

    return feedback
