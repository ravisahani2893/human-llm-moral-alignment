def build_prompt(situation):

    return f"""
You are an expert evaluator of moral situations.

Your task is to independently evaluate TWO aspects of the situation.

1. Action Moral Valence
2. Consequence Moral Valence

Definitions:

• Action Moral Valence refers ONLY to the morality of the action performed, regardless of its outcome.

• Consequence Moral Valence refers ONLY to the morality of the consequence or outcome resulting from that action, regardless of whether the action itself was morally right or wrong.

Use a continuous numerical scale between -1.0 and +1.0.

Scale:

-1.00 = Extremely morally negative
-0.75 = Strongly morally negative
-0.50 = Moderately morally negative
-0.25 = Slightly morally negative
 0.00 = Morally neutral
 0.25 = Slightly morally positive
 0.50 = Moderately morally positive
 0.75 = Strongly morally positive
 1.00 = Extremely morally positive

Important Instructions:

• Think carefully before assigning the scores.
• Evaluate the action and the consequence independently.
• The action and consequence may receive different scores.
• Use the entire range of the scale.
• Do NOT restrict yourself to only -1, 0 or 1.
• Decimal values such as -0.83, -0.41, 0.18, 0.63 and 0.91 are expected whenever appropriate.
• Only assign 0.00 if the action or consequence is genuinely morally neutral.

Return ONLY a valid JSON object.

Do not include markdown.
Do not include explanations.
Do not include any additional text.

JSON format:

{{
    "action_valence": <float between -1.0 and 1.0>,
    "consequence_valence": <float between -1.0 and 1.0>
}}

Situation:

{situation}
"""