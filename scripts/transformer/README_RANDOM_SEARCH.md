# Transformer random search

The campaign is launched with one command and no command-line parameters:

```bash
sbatch scripts/transformer/random_search_and_evaluate.sh
```

All parameters are defined near the top of
`scripts/transformer/random_search_and_evaluate.sh`. Set `DRY_RUN=true` there
to generate and validate the screening configurations without training.

## Protocol

- Trial `000` is the current reference configuration.
- The other trials sample the temporal window, Transformer architecture,
  positional encoding, learning rate, and AdamW weight decay.
- Loss weights and input features remain fixed. The competitive campaign uses
  pitch points and ground intersection, while the valid-joint mask and root
  initialization as an explicit model feature are disabled.
- Screening uses all training matches, with a capped number of windows per
  sequence, and ranks candidates only on the validation match.
- The test match is not read for model selection.
- The best candidate is retrained from scratch on every training window and is
  then evaluated on both validation and test.

The Slurm wall time is 32 hours. The Python orchestrator uses an internal
31-hour deadline, limits each screening trial to two hours, and reserves ten
hours for full retraining and final evaluation. If screening runs more slowly
than expected, remaining candidates are skipped instead of consuming the final
training reserve.

## Outputs

For the default search name, campaign files are written under:

```text
outputs/ablation/unseen_match_v1/random_search/root_transformer_random_search_v2_competitive/
```

Important files are:

```text
random_search_results.csv
random_search_results_ranked.csv
best_screening_trial.json
best_screening_config.yaml
best_config.yaml
random_search_summary.json
logs/
```

The final model uses a run name of the form
`root_transformer_random_search_v2_competitive_best_trial_XXX_full`. Its checkpoint,
predictions, and metrics follow the standard project layout under
`outputs/ablation/unseen_match_v1/{checkpoints,predictions,eval_reports}/`.

The campaign is resumable. Submitting the same shell script again reuses trials
and final outputs that already contain a checkpoint and training summary. To
start an independent campaign, change `SEARCH_NAME` in the shell script.

## Retrain the top three at full budget

The screening ranking can be consolidated by training its top three candidates
on all training windows:

```bash
sbatch scripts/transformer/retrain_random_search_top3.sh
```

This launches three sequential full-budget trainings with the same seed. The
winner is selected using only the validation root error; only that selected
checkpoint is then evaluated on the test match. Parameters, including dry-run
mode and the 32-hour wall-time budget, are defined directly in the shell script.

Outputs are written under:

```text
outputs/ablation/unseen_match_v1/random_search/root_transformer_random_search_v2_competitive/full_budget_top3/
```
