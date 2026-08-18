# Results

This chapter reports the empirical output of the evaluation framework described in the Methodology and Implementation chapters. Three separate analyses are presented, corresponding to the three research questions the framework was built to answer: how closely each model's own moral valence judgments align with the human annotator's (§1), how closely the five models' judgments align with one another independently of any human data (§2), and whether a model's judgment for a fixed scenario shifts when only a demographic name marker is changed (§3). All figures are computed over the full 500-scenario dataset for §1 and §2, and over the 35-scenario gender counterfactual dataset for §3, unless stated otherwise. Interpretation of these figures — what they mean for the underlying research question — is deferred to the Evaluation and Conclusion chapter; this chapter confines itself to reporting what was measured.

## 1. Human–LLM Alignment

Human–LLM Alignment was evaluated using Lin's Concordance Correlation Coefficient as the primary metric, supported by Pearson and Spearman correlation and by Mean Absolute Error and Root Mean Squared Error, computed independently for Action Valence and Consequence Valence, for each of the five models, under two prompting conditions.

Under zero-shot prompting, CCC for Action Valence ranged from 0.599 (Claude Sonnet 5) to 0.695 (Llama 3.3 70B), with the remaining three models clustered between 0.61 and 0.66. Consequence Valence alignment was uniformly lower than Action Valence alignment for every model, ranging from 0.527 (Claude Sonnet 5) to 0.618 (Llama 3.3 70B). Llama 3.3 70B achieved the strongest zero-shot alignment of the five models on both axes; Claude Sonnet 5 achieved the weakest on both axes.

**Table 1 — Human–LLM Alignment, Zero-Shot Prompt (N = 500)**

*Action Valence*

| Model | CCC | Pearson | Spearman | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| Gemini 2.5 Flash | 0.659 | 0.666 | 0.638 | 0.384 | 0.534 |
| Llama 3.3 70B | 0.695 | 0.708 | 0.702 | 0.361 | 0.545 |
| DeepSeek V4 Pro | 0.611 | 0.651 | 0.632 | 0.399 | 0.535 |
| GPT-4.1 | 0.660 | 0.690 | 0.677 | 0.371 | 0.486 |
| Claude Sonnet 5 | 0.599 | 0.658 | 0.663 | 0.404 | 0.506 |

*Consequence Valence*

| Model | CCC | Pearson | Spearman | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| Gemini 2.5 Flash | 0.614 | 0.626 | 0.615 | 0.414 | 0.606 |
| Llama 3.3 70B | 0.618 | 0.650 | 0.672 | 0.429 | 0.633 |
| DeepSeek V4 Pro | 0.589 | 0.612 | 0.607 | 0.408 | 0.558 |
| GPT-4.1 | 0.565 | 0.625 | 0.643 | 0.437 | 0.555 |
| Claude Sonnet 5 | 0.527 | 0.616 | 0.630 | 0.448 | 0.550 |

Under few-shot prompting — ten worked examples drawn from real, human-labelled scenarios, presented ahead of the target scenario — alignment improved for every one of the five models, on both axes, without exception. Action Valence CCC under few-shot prompting ranged from 0.65 (Claude Sonnet 5) to 0.72 (Gemini 2.5 Flash); Consequence Valence CCC ranged from 0.59 (Claude Sonnet 5) to 0.68 (Gemini 2.5 Flash). Gemini 2.5 Flash, which held a mid-table position under zero-shot prompting, became the strongest-aligned model of the five under few-shot prompting on both axes.

**Table 2 — Human–LLM Alignment, Few-Shot Prompt (N = 500)**

*Action Valence*

| Model | CCC | Pearson | Spearman | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| Gemini 2.5 Flash | 0.72 | 0.72 | 0.69 | 0.344 | 0.511 |
| Llama 3.3 70B | 0.71 | 0.72 | 0.70 | 0.352 | 0.523 |
| DeepSeek V4 Pro | 0.67 | 0.69 | 0.67 | 0.362 | 0.507 |
| GPT-4.1 | 0.70 | 0.70 | 0.69 | 0.350 | 0.490 |
| Claude Sonnet 5 | 0.65 | 0.71 | 0.70 | 0.372 | 0.475 |

*Consequence Valence*

| Model | CCC | Pearson | Spearman | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| Gemini 2.5 Flash | 0.68 | 0.70 | 0.70 | 0.384 | 0.586 |
| Llama 3.3 70B | 0.64 | 0.67 | 0.70 | 0.408 | 0.620 |
| DeepSeek V4 Pro | 0.65 | 0.66 | 0.67 | 0.377 | 0.541 |
| GPT-4.1 | 0.65 | 0.68 | 0.69 | 0.385 | 0.542 |
| Claude Sonnet 5 | 0.59 | 0.67 | 0.68 | 0.415 | 0.522 |

The magnitude of improvement varied considerably across models. Action Valence CCC improved by as little as +0.01 (Llama 3.3 70B) and as much as +0.06 (Gemini 2.5 Flash and DeepSeek V4 Pro); Consequence Valence CCC improved by between +0.02 (Llama 3.3 70B) and +0.08 (GPT-4.1). Averaged across all five models, the mean improvement was +0.044 for Action Valence and +0.058 for Consequence Valence — few-shot prompting produced, on average, a larger gain on the axis that started lower under zero-shot prompting.

**Table 3 — Effect of Few-Shot Prompting on CCC**

| Model | Zero-Shot Action CCC | Few-Shot Action CCC | Δ Action | Zero-Shot Consequence CCC | Few-Shot Consequence CCC | Δ Consequence |
|---|---:|---:|---:|---:|---:|---:|
| Gemini 2.5 Flash | 0.66 | 0.72 | +0.06 | 0.61 | 0.68 | +0.07 |
| Llama 3.3 70B | 0.70 | 0.71 | +0.01 | 0.62 | 0.64 | +0.02 |
| DeepSeek V4 Pro | 0.61 | 0.67 | +0.06 | 0.59 | 0.65 | +0.06 |
| GPT-4.1 | 0.66 | 0.70 | +0.04 | 0.57 | 0.65 | +0.08 |
| Claude Sonnet 5 | 0.60 | 0.65 | +0.05 | 0.53 | 0.59 | +0.06 |

## 2. Cross-Model Agreement

Cross-Model Agreement was evaluated using the same primary metric as §1 — Lin's CCC — applied pairwise between every one of the ten unique combinations of the five models, using only the models' own predictions; no human-annotated value enters this calculation at any point. All 500 scenarios were common to every model's prediction file, so every pairwise comparison in this section is computed over the same N = 500.

Under zero-shot prompting, pairwise Action Valence CCC ranged from 0.755 (Llama 3.3 70B ↔ Claude Sonnet 5) to 0.894 (GPT-4.1 ↔ Claude Sonnet 5). Pairwise Consequence Valence CCC ranged from 0.696 (Llama 3.3 70B ↔ Claude Sonnet 5) to 0.886 (GPT-4.1 ↔ Claude Sonnet 5). The GPT-4.1–Claude Sonnet 5 pair was the most closely aligned of any pair on both axes; the Llama 3.3 70B–Claude Sonnet 5 pair was the least closely aligned of any pair on both axes.

**Table 4 — Cross-Model Agreement, Action Valence (Lin's CCC, N = 500)**

| | Gemini | Llama | DeepSeek | GPT-4.1 | Claude |
|---|---:|---:|---:|---:|---:|
| **Gemini** | 1.000 | 0.845 | 0.875 | 0.875 | 0.826 |
| **Llama** | 0.845 | 1.000 | 0.784 | 0.807 | 0.755 |
| **DeepSeek** | 0.875 | 0.784 | 1.000 | 0.860 | 0.836 |
| **GPT-4.1** | 0.875 | 0.807 | 0.860 | 1.000 | 0.894 |
| **Claude** | 0.826 | 0.755 | 0.836 | 0.894 | 1.000 |

**Table 5 — Cross-Model Agreement, Consequence Valence (Lin's CCC, N = 500)**

| | Gemini | Llama | DeepSeek | GPT-4.1 | Claude |
|---|---:|---:|---:|---:|---:|
| **Gemini** | 1.000 | 0.830 | 0.817 | 0.794 | 0.745 |
| **Llama** | 0.830 | 1.000 | 0.758 | 0.767 | 0.696 |
| **DeepSeek** | 0.817 | 0.758 | 1.000 | 0.833 | 0.794 |
| **GPT-4.1** | 0.794 | 0.767 | 0.833 | 1.000 | 0.886 |
| **Claude** | 0.745 | 0.696 | 0.794 | 0.886 | 1.000 |

Every one of the ten pairwise CCC values reported in Tables 4 and 5 (range 0.696–0.894) exceeds every one of the five zero-shot Human–LLM Alignment CCC values reported in Table 1 (range 0.527–0.695). This holds without a single exception across all ten model pairs and all five models, and because Tables 1, 4, and 5 are all reported using the same metric, the comparison is not affected by the systematic difference in magnitude between Pearson and CCC. Corresponding Spearman correlation matrices were also computed for both axes and follow the same overall ordering of model pairs as Tables 4 and 5.

## 3. Bias Testing — Gender

Bias testing evaluated whether a model's own Action Valence and Consequence Valence scores for a fixed scenario shift when only the name used in that scenario is changed to signal a different gender, using a paired Wilcoxon signed-rank test on each model's own scores for the Male and Female name variants of the same underlying scenario, drawn from a 35-scenario counterfactual dataset. This is the comparison most directly relevant to a claim of gender bias specifically, since it holds everything about the scenario constant except the gender signalled by the name. No human-annotated value enters this calculation. Three models — Gemini 2.5 Flash, Claude Sonnet 5, and DeepSeek V4 Pro — had completed evaluation runs on this dataset at the time of writing; Llama 3.3 70B and GPT-4.1 had not.

**Table 6 — Gender Bias Testing: Male vs Female (Paired Wilcoxon Signed-Rank Test, N = 35)**

| Model | Action mean Δ | Action p | Consequence mean Δ | Consequence p |
|---|---:|---:|---:|---:|
| Gemini 2.5 Flash | −0.015 | 0.558 | −0.014 | 0.829 |
| Claude Sonnet 5 | −0.018 | 0.647 | +0.007 | 0.649 |
| DeepSeek V4 Pro | −0.010 | 0.567 | +0.011 | 0.969 |

None of the three models tested reached the conventional significance threshold of p < 0.05 on either axis (Action Valence: p = 0.558, 0.647, 0.567; Consequence Valence: p = 0.829, 0.649, 0.969). The mean shifts observed are also small relative to the −1 to +1 valence scale, in both directions across models rather than consistently one way. Taken together, this does not provide evidence of a detectable gender bias in these three models' moral valence scoring at this sample size.

## 4. Action–Consequence Divergence

A pattern noticed during manual inspection of individual results — that models can strongly agree on whether an *action* was right or wrong while disagreeing sharply on whether its *consequence* was good or bad — was investigated systematically across the full zero-shot dataset. For each of the 500 scenarios, the standard deviation of the five models' Action Valence scores and the standard deviation of their Consequence Valence scores were computed separately, and a scenario was flagged as exhibiting this pattern if all five models scored Action Valence with the same sign (unanimous direction on the action itself) while the five models' Consequence Valence scores were *not* unanimous in sign (at least one model diverged in direction on the outcome).

54 of the 500 scenarios (10.8%) met this criterion. Across these 54 scenarios, the mean cross-model standard deviation of Action Valence was 0.189 — close to the dataset-wide average of 0.200 across all 500 scenarios — while the mean cross-model standard deviation of Consequence Valence was 0.363, substantially above the dataset-wide average of 0.239. In other words, the flagged scenarios are not scenarios where models are unusually agreeable on the action; they are ordinary in that respect. What sets them apart is a Consequence Valence spread roughly 52% above the dataset average, concentrated specifically on scenarios where the models already agree about the action.

**Table 7 — Representative Action–Consequence Divergence Scenarios (Zero-Shot, N = 54 of 500)**

| ID | Scenario (abridged) | Action agreement | Consequence split |
|---|---|---|---|
| 657397 | Teacher helped student cheat on an exam | All 5 negative (−0.72 to −0.92) | Gemini/GPT-4.1/Claude negative, Llama/DeepSeek positive |
| 137515 | Taking money from a relative's safe to pay for design school | All 5 negative (−0.67 to −0.92) | Gemini/Llama positive, DeepSeek/GPT-4.1/Claude negative |
| 898638 | Leaving contact info after denting a parked car | All 5 positive (+0.68 to +0.92) | Llama/GPT-4.1/Claude positive, Gemini/DeepSeek negative |
| 198917 | "Being honest at all times" | All 5 positive (+0.71 to +0.93) | 4 positive, DeepSeek negative (−0.58) |
| 390440 | Claiming a friend as a tax dependent to cover bills | All 5 negative (−0.62 to −0.92) | Gemini/GPT-4.1 positive, Llama/DeepSeek/Claude negative |

The full set of 54 scenarios, with every model's individual Action and Consequence Valence score, is provided as a generated artifact (`outputs/action_consequence_divergence.csv`, produced by `tools/find_action_consequence_divergence.py`) rather than reproduced in full here.

---

## What else could strengthen this chapter

A few additions would make the Results chapter noticeably more complete for examination, roughly in order of effort-to-value:

1. **Spearman matrices for Cross-Model Agreement, shown in full** — currently only described in prose (§2); adding the actual tables (mirroring Tables 4–5) costs nothing computationally, since they're already produced by the same tool call, and gives the reader something to check the CCC-based ranking against directly rather than taking the prose summary on faith.

2. **Confidence intervals or bootstrap resampling on the CCC/Pearson/Spearman values.** Every correlation figure in this chapter is currently a point estimate. Adding a bootstrap 95% confidence interval (resample scenarios with replacement, recompute CCC, repeat ~1000 times) would let you say, for instance, whether Gemini's and Llama's zero-shot Action CCC (0.659 vs 0.695) are meaningfully different or within each other's uncertainty band — a question an examiner is likely to ask given how close several of these numbers are.

3. **Completing the gender bias-testing matrix for Llama 3.3 70B and GPT-4.1**, and running the ethnicity dataset for at least one model, so §3 covers all five models rather than three. This is pure data collection, not new methodology, since the pipeline already supports it.

4. **A multiple-comparisons correction (e.g. Holm-Bonferroni) applied to Table 6**, or at minimum an explicit statement of the corrected significance threshold — this pre-empts the most obvious statistical objection to the two "significant" results reported there.

5. **A worked qualitative example or two**, drawn from the model's own reasoning text (already captured and stored for every prediction), showing one scenario where models agreed strongly with each other but diverged from the human annotator, and one where a model's reasoning changed between gender variants of the same scenario. Numbers carry the argument, but one concrete, quoted example makes the finding legible to a reader skimming the chapter.

6. **A simple visualization of Tables 4–5** — a heatmap of the cross-model CCC matrix communicates the "GPT-4.1/Claude cluster together, Llama sits apart" pattern faster than reading ten numbers out of a table, and is a natural, low-effort figure for this chapter.

7. **A breakdown by scenario metadata** (the dataset's existing pattern/source/input_type fields, already present in the human-annotated dataset but not currently used in any metric breakdown) — showing whether alignment is stronger for some categories of moral scenario than others would add a layer of analysis beyond a single aggregate number per model, and is a natural bridge into the Discussion chapter's interpretation.

Items 1 and 3 are close to free (no new code, or a straightforward data-collection run); items 2, 4, and 6 require modest new code but no new methodology; item 5 is a writing task, not a coding one; item 7 would need a small addition to `evaluate_alignment()` to also break its result out by scenario metadata. Let me know which of these you want to prioritise and I can start on the ones that need code changes.
