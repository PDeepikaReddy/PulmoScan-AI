"""
Educational guidance shown alongside a prediction.

IMPORTANT: This content is for general awareness only and is NOT a medical
diagnosis. It must never replace evaluation by a qualified clinician.
"""

TUBERCULOSIS = {
    "summary": (
        "The model flagged radiographic patterns often associated with "
        "pulmonary tuberculosis (TB). TB is a treatable and curable bacterial "
        "infection, but it requires confirmation and supervised treatment by a "
        "doctor."
    ),
    "immediate_steps": [
        "Consult a pulmonologist or visit a TB/DOTS center as soon as possible.",
        "Get confirmatory tests: sputum smear microscopy, GeneXpert (CB-NAAT), "
        "and a clinical chest X-ray review.",
        "Until you are evaluated, wear a mask and cover your mouth when coughing "
        "to protect those around you.",
        "Inform close household contacts so they can also be screened.",
    ],
    "treatment": [
        "Standard TB treatment is a 6-month course of antibiotics (often "
        "isoniazid, rifampicin, pyrazinamide, ethambutol).",
        "Complete the FULL course even after you feel better — stopping early "
        "causes drug-resistant TB.",
        "Treatment is provided free of charge under national TB programmes in "
        "many countries (e.g., India's Ni-kshay / DOTS).",
    ],
    "lifestyle": [
        "Eat a protein- and calorie-rich diet (eggs, lentils, milk, nuts, lean "
        "meat) to rebuild strength.",
        "Ensure good ventilation and sunlight in living spaces.",
        "Avoid smoking and alcohol, which worsen lung damage and interfere with "
        "medication.",
        "Rest adequately and stay hydrated.",
        "Get adequate vitamin D and a daily multivitamin if advised by a doctor.",
    ],
    "warning_signs": [
        "Coughing up blood",
        "Severe breathlessness or chest pain",
        "High fever that does not subside",
        "Rapid, unexplained weight loss",
    ],
    "disclaimer": (
        "This is an AI screening aid, not a diagnosis. Please confirm with a "
        "qualified physician and laboratory tests."
    ),
}

NORMAL = {
    "summary": (
        "The model did not detect radiographic patterns typically associated "
        "with tuberculosis in this chest X-ray. This is reassuring, but it is "
        "not a substitute for a doctor's interpretation."
    ),
    "immediate_steps": [
        "If you have symptoms (persistent cough > 2 weeks, fever, night sweats, "
        "weight loss), still see a doctor — early TB or other conditions can be "
        "subtle.",
        "Keep this X-ray on record for future comparison.",
    ],
    "treatment": [
        "No TB-specific treatment is indicated by this screening result.",
        "Follow up on any ongoing symptoms with a general physician.",
    ],
    "lifestyle": [
        "Maintain a balanced diet and regular exercise to keep your lungs "
        "healthy.",
        "Avoid smoking and limit exposure to air pollution and dust.",
        "Stay up to date with routine health check-ups.",
        "Practice good respiratory hygiene, especially in crowded places.",
    ],
    "warning_signs": [
        "A new cough lasting more than two weeks",
        "Unexplained weight loss or persistent tiredness",
        "Recurring low-grade fever or night sweats",
    ],
    "disclaimer": (
        "An AI screening result of 'Normal' does not rule out all disease. "
        "Consult a physician if symptoms persist."
    ),
}


def get_guidance(label):
    """Return the guidance dict for a predicted label."""
    return TUBERCULOSIS if label == "Tuberculosis" else NORMAL
