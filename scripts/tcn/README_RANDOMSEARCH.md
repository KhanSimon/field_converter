# Random search TCN

## Lancer une recherche

Depuis la racine du projet:

```bash
sbatch scripts/tcn/random_search.sh
```

ou en direct:

```bash
PYTHONPATH=src python scripts/tcn/random_search.py \
  --base-config configs/tcn/root_tcn_v1.yaml \
  --search-name root_tcn_random_search \
  --n-trials 12 \
  --seed 1234 \
  --output-dir outputs \
  --skip-existing
```

Les essais sont sauvegardes dans:

```text
outputs/random_search/<search-name>/
outputs/checkpoints/<search-name>_trial_XXX/
outputs/eval_reports/<search-name>_trial_XXX/
outputs/predictions/<search-name>_trial_XXX/
```

## Modifier les hyperparametres testes

Les hyperparametres tires au hasard sont dans:

```text
scripts/tcn/random_search.py
```

fonction:

```python
sample_trial(rng)
```

Pour retirer un hyperparametre du random search, supprimer simplement sa ligne dans le `return` de `sample_trial`.

Exemple:

```python
"model.dropout": round_float(rng.uniform(0.0, 0.35), 5),
```

Si on supprime cette ligne, `dropout` gardera la valeur du fichier de base:

```text
configs/tcn/root_tcn_v1.yaml
```

Pour modifier une plage, changer seulement la distribution:

```python
"optimizer.lr": round_float(log_uniform(rng, 1e-4, 1e-3), 8),
"dataset.window_size": rng.choice([61, 81, 101]),
"model.dropout": round_float(rng.uniform(0.05, 0.25), 5),
```

Attention: `loss_weights.root_axis_weights` est aussi decompose en colonnes `loss_weights.root_axis_x/y/z` pour les plots. Si on retire `root_axis_weights`, ces colonnes ne seront plus ajoutees, ce qui est OK.

## Refaire une nouvelle recherche

Changer `--search-name` dans:

```text
scripts/tcn/random_search.sh
```

Exemple:

```bash
--search-name root_tcn_random_search_v2
```

Chaque `search-name` cree une campagne separee, avec ses propres configs et resultats.

Si on garde le meme `search-name`, les nouveaux trials seront melanges dans le meme dossier. Avec `--skip-existing`, un trial deja entraine ne sera pas relance si son `best.pt` existe.

Pour continuer une campagne sans refaire les memes tirages, utiliser:

```bash
--start-index 12 --n-trials 12
```

Par exemple, apres les trials `000` a `011`, cette commande lance `012` a `023`.

## Evaluer la recherche

Apres les entrainements:

```bash
sbatch scripts/tcn/evaluate_random_search.sh
```

ou:

```bash
PYTHONPATH=src python scripts/tcn/evaluate_random_search.py \
  --search-name root_tcn_random_search \
  --output-dir outputs \
  --metric best_root_error_mean_m \
  --top-k 8
```

Le script choisit le meilleur run selon `best_root_error_mean_m`, puis relance l'evaluation du meilleur checkpoint.

Plots produits pour la recherche:

```text
outputs/random_search/<search-name>/plots/score_by_trial.png
outputs/random_search/<search-name>/plots/top_trials.png
outputs/random_search/<search-name>/plots/hyperparameter_scatters.png
```

Plots produits pour le meilleur run:

```text
outputs/eval_reports/<best-run>/plots/training_curves.png
outputs/eval_reports/<best-run>/plots/reproj_overlay_valid.png
outputs/eval_reports/<best-run>/plots/root_xyz_timeseries_valid.png
outputs/eval_reports/<best-run>/plots/world_traj_xy_valid.png
```
