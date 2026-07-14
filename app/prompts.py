def build_prompt(situation):

    return f"""
You are an expert evaluator of moral situations.

Your task is to independently evaluate TWO aspects of the situation.

1. Action Moral Valence
2. Consequence Moral Valence



Definitions:

• Action Moral Valence refers ONLY to the morality of the action performed, regardless of its outcome.

• Consequence Moral Valence refers ONLY to the morality of the consequence or outcome resulting from that action, regardless of whether the action itself was morally right or wrong.

Use a continuous numerical scale from -1.0 (extremely morally negative)
through 0.0 (morally neutral) to +1.0 (extremely morally positive).

This is a CONTINUOUS scale, not a small set of fixed steps. Treat it like a
thermometer reading, not a multiple-choice menu.

Important Instructions:

• Think carefully before assigning the scores.
• Evaluate the action and the consequence independently.
• The action and consequence may receive different scores.
• Score to two decimal places, reflecting the precise intensity you perceive.
• Do NOT round to the nearest 0.25 or 0.5. Avoid "clean" numbers like
  -0.25, -0.50, -0.75, 0.25, 0.50, 0.75 unless the intensity is genuinely
  exactly that strong — most real judgments will NOT land on those exact
  values.
• Only assign 0.00 if the action or consequence is genuinely morally neutral.

Examples of the level of precision expected: -0.87, -0.62, -0.41, -0.19,
0.08, 0.33, 0.57, 0.79, 0.94.

In addition to each score, provide:

• A short reasoning (1-2 sentences) explaining why you assigned that score,
  written for someone who cannot see the scenario and needs the score
  justified on its own.
• A list of 2-4 short key factors (a few words each, not full sentences)
  that most influenced that score.

Return ONLY a valid JSON object.

Do not include markdown.
Do not include any additional text outside the JSON object.

JSON format:

{{
    "action_valence": <float between -1.0 and 1.0>,
    "action_reasoning": "<1-2 sentence justification>",
    "action_factors": ["<short factor>", "<short factor>"],
    "consequence_valence": <float between -1.0 and 1.0>,
    "consequence_reasoning": "<1-2 sentence justification>",
    "consequence_factors": ["<short factor>", "<short factor>"]
}}

Situation:

{situation}
"""