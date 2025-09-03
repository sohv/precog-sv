"""
Evaluation Prompts for Persona Vector Validation

This file contains held-out prompts that are NOT used for vector extraction,
only for testing the generalization of extracted persona vectors.

These prompts are semantically different from the extraction prompts to ensure
proper validation without data leakage.
"""

EVALUATION_PROMPTS = {
    "openness": {
        "high": [
            "Plan a vacation that involves exploring completely unfamiliar territories and cultures.",
            "Design your ideal workspace that encourages innovative thinking.",
            "Describe how you would approach learning a skill that seems impossible at first.",
            "What would you do if you had unlimited resources to solve a global problem?",
            "Imagine you're designing a new city from scratch - what would it look like?",
            "How would you spend a day if there were no rules or expectations?",
            "Describe your dream job that doesn't exist yet.",
            "What's the most unconventional book or movie you'd recommend and why?",
        ],
        "low": [
            "Plan a vacation to a familiar, comfortable destination with predictable activities.",
            "Describe your preferred workspace that follows traditional office layouts.",
            "How do you approach learning something by following established, proven methods?",
            "What reliable, time-tested solution would you apply to a common problem?",
            "Describe an ideal neighborhood that follows conventional community standards.",
            "How would you spend a day following a well-structured, familiar routine?",
            "Describe a stable, traditional career path you find appealing.",
            "What classic book or movie do you think everyone should read/watch and why?",
        ]
    },
    "extraversion": {
        "high": [
            "Describe how you would energize a team that's feeling unmotivated.",
            "Plan an event that brings together people from different backgrounds.",
            "How would you approach networking at a professional conference?",
            "Describe your ideal way to celebrate a personal achievement.",
            "How would you handle being the new person in a large, active workplace?",
            "What's your approach to making friends in a new city?",
            "Describe how you'd lead a group discussion on a controversial topic.",
            "How would you spend your ideal weekend with complete freedom to choose?",
        ],
        "low": [
            "Describe how you would motivate yourself when working alone on a project.",
            "Plan a meaningful one-on-one conversation with someone important to you.",
            "How would you prepare for a required social professional event?",
            "Describe your preferred way to process and reflect on personal accomplishments.",
            "How would you adjust to working in a quiet, independent role?",
            "What's your approach to building deeper connections with existing friends?",
            "Describe how you'd contribute thoughtfully to a small group discussion.",
            "How would you spend your ideal weekend with time for solitude and reflection?",
        ]
    },
    "conscientiousness": {
        "high": [
            "Describe your approach to managing a complex, long-term project with multiple deadlines.",
            "How would you organize your home to maximize efficiency and productivity?",
            "Plan a study schedule for learning a challenging new subject over six months.",
            "Describe how you would prepare for an important presentation to senior executives.",
            "How would you handle competing priorities when everything seems urgent?",
            "What's your method for maintaining quality standards under time pressure?",
            "Describe how you would train someone to take over your responsibilities.",
            "How would you approach saving money for a major future purchase?",
        ],
        "low": [
            "Describe how you handle projects when inspiration strikes rather than following rigid schedules.",
            "How would you organize your living space to feel comfortable and lived-in?",
            "Plan an exploratory approach to learning something new without strict timelines.",
            "Describe how you would give an authentic, spontaneous presentation.",
            "How do you navigate competing demands by staying flexible and adaptive?",
            "What's your approach to maintaining good-enough standards while staying relaxed?",
            "Describe how you would share your knowledge in an informal, collaborative way.",
            "How would you approach financial decisions by staying open to opportunities?",
        ]
    },
    "agreeableness": {
        "high": [
            "Describe how you would mediate a conflict between two close friends.",
            "How would you approach giving feedback to someone who's sensitive to criticism?",
            "Plan a group activity that ensures everyone feels included and valued.",
            "Describe how you would handle a situation where someone needs help but hasn't asked.",
            "How would you respond if someone took credit for your work?",
            "What's your approach to supporting a teammate who's struggling with their tasks?",
            "Describe how you would welcome a new person to your friend group or workplace.",
            "How would you handle a disagreement with someone you care about?",
        ],
        "low": [
            "Describe how you would address a situation where someone is consistently underperforming.",
            "How would you give direct, honest feedback when someone needs to improve?",
            "Plan a competitive activity where excellence and winning are the primary goals.",
            "Describe how you would prioritize your own needs when others are making demands.",
            "How would you respond if someone was taking advantage of your efforts?",
            "What's your approach to maintaining professional boundaries with colleagues?",
            "Describe how you would handle someone who's not contributing their fair share.",
            "How would you stand firm on an important principle during a disagreement?",
        ]
    },
    "neuroticism": {
        "high": [
            "Describe how you experience and process stressful situations at work.",
            "How do you handle uncertainty when facing major life decisions?",
            "What goes through your mind when you receive unexpected criticism?",
            "Describe your response to situations where you feel overwhelmed.",
            "How do you cope when things don't go according to your plans?",
            "What's your experience with worry about future events or outcomes?",
            "Describe how you handle situations where you feel judged by others.",
            "How do you respond when you're facing multiple pressures simultaneously?",
        ],
        "low": [
            "Describe how you maintain calm and perspective during challenging periods.",
            "How do you confidently navigate uncertain situations and unknown outcomes?",
            "What's your typical response to constructive feedback or criticism?",
            "Describe how you stay balanced when facing demanding circumstances.",
            "How do you adapt positively when your original plans need to change?",
            "What's your approach to staying optimistic about future possibilities?",
            "Describe how you remain confident in social or evaluative situations.",
            "How do you maintain emotional stability when dealing with multiple challenges?",
        ]
    }
}

def get_evaluation_prompts(trait: str, high_behavior: bool = True, n_samples: int = 5):
    """
    Get evaluation prompts for testing persona vector generalization.
    
    Args:
        trait: Personality trait ('openness', 'extraversion', etc.)
        high_behavior: If True, return high-trait prompts; if False, low-trait prompts
        n_samples: Number of prompts to return
        
    Returns:
        List of evaluation prompts
    """
    if trait.lower() not in EVALUATION_PROMPTS:
        raise ValueError(f"Trait '{trait}' not supported. Available: {list(EVALUATION_PROMPTS.keys())}")
    
    condition = "high" if high_behavior else "low"
    prompts = EVALUATION_PROMPTS[trait.lower()][condition]
    
    return prompts[:n_samples]