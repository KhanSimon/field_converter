# Réentraînement des références du papier sur ARG_FRA

Les 9 séquences d'ARG_FRA appartiennent au train du split d'ablation `primary`.
Pour évaluer ce match comme un match inconnu, il faut repartir de zéro avec
un nouveau split et recalculer les statistiques de normalisation sur son train.
Changer uniquement le split d'évaluation d'un ancien checkpoint ne suffit pas.

| Split | Matchs | Séquences |
|---|---|---:|
| Train | ARG_CRO, CRO_MOR, ENG_FRA, FRA_MOR, MOR_POR, NET_ARG | 68 |
| Validation | BRA_KOR | 12 |
| Test | ARG_FRA | 9 |

ENG_FRA revient donc dans le train. Les deux runs reprennent exactement les
hyperparamètres des YAML `configs/paper/`, y compris les seeds :

| Tâche Slurm | Modèle | Fenêtre | Seed | Run source |
|---|---|---:|---:|---|
| 0 | TCN | 41 | 2027 | `primary_tcn_full_s2027` |
| 1 | Transformer | 41 | 1235 | `primary_transformer_window_41_s1235` |

Seuls les noms et chemins des nouveaux runs changent. La sélection du checkpoint
et l'early stopping utilisent BRA_KOR ; ARG_FRA est évalué après le train.

## Lancement

Depuis la racine du dépôt :

```bash
mkdir -p slurms
sbatch scripts/arg_fra/submit_train.sh
```

Le submitter génère les deux configurations puis soumet la préparation CPU.
La dépendance `afterok` attend sa réussite avant de lancer l'array `0-1%1` :
un seul GPU L40S est utilisé à la fois. Chaque tâche entraîne son modèle puis
évalue `best.pt` sur validation et test, en sauvegardant les prédictions NPZ.

Le prétraitement réutilise le code d'ablation : features brutes de `data/features`,
pelvis `hips_mean`, normalisation des features et des roots initiaux ajustée
uniquement sur les 68 séquences du train. Les roots initiaux bruts de
`data/root_init_cam` sont réutilisés si leur provenance est compatible ; le
préparateur existant les régénère sinon.

Les ressources Slurm et options de reprise sont en tête des scripts shell.
L'environnement `cv_train_clean` et les temporaires courts proviennent de
`scripts/ablation/common_env.sh`.

## Vérification sans entraînement

Dans un environnement Python avec les dépendances du projet :

```bash
PYTHONPATH=src python scripts/arg_fra/run.py submit --dry-run
```

Cette commande vérifie les fichiers de features, génère les deux YAML et le
plan avec la liste exacte des séquences, puis affiche les commandes Slurm.
Elle ne soumet aucun job et ne calcule pas les données normalisées.
On peut aussi définir `DRY_RUN=true` dans `submit_train.sh` avant de le soumettre.

Les tests sur des clips synthétiques vérifient la normalisation sans fuite,
la conservation des hyperparamètres, les dépendances Slurm et la reprise.
Ils s'exécutent sans GPU dans `cv_train_clean` :

```bash
PYTHONPATH=src python -m unittest discover -s scripts/arg_fra -p 'test_*.py' -v
```

## Fichiers produits

```text
data/match_tests/arg_fra_v1/
  preprocessing_complete.json
  arg_fra/
    features_normalized/{split.json,normalization_stats.*,train/,valid/,test/}
    root_init_cam_normalized/{split.json,root_init_normalization_meta.json,train/,valid/,test/}

outputs/match_tests/arg_fra_v1/
  manifest_used.yaml
  plan.json
  submission.json
  submission_dry_run.json
  configs/arg_fra_{tcn_s2027,transformer_s1235}.yaml
  checkpoints/<run_name>/{best.pt,last.pt}
  eval_reports/<run_name>/{config_used.yaml,train_summary.json,metrics.json}
  predictions/<run_name>/{valid_predictions.npz,test_predictions.npz}
  states/<run_name>.json
```

Les résultats et données normalisées des anciennes campagnes restent dans leurs
dossiers. Les nouveaux noms de runs sont `arg_fra_tcn_s2027` et
`arg_fra_transformer_s1235`.

## Reprise et exécution par étapes

Relancer `sbatch scripts/arg_fra/submit_train.sh` réutilise les données préparées
et saute les runs terminés. Un train avec `best.pt` et `train_summary.json`
reprend à l'évaluation ; un train interrompu sans résumé repart de zéro.
`FORCE_TRAIN` et `FORCE_EVAL` se règlent dans `train.sh`.

Pour préparer uniquement les données :

```bash
sbatch scripts/arg_fra/prepare.sh
```

Une fois ce job terminé avec succès, lancer les deux modèles, ou un seul :

```bash
sbatch scripts/arg_fra/train.sh
sbatch --array=0 scripts/arg_fra/train.sh  # TCN uniquement
sbatch --array=1 scripts/arg_fra/train.sh  # Transformer uniquement
```

Choisir une seule de ces trois commandes. Avant chaque train, le script vérifie
le plan, les splits préparés et la liste des séquences ayant servi aux statistiques.
Les configs sont figées dès leur génération : pour changer le split ou les
hyperparamètres, choisir un nouveau `campaign_name` dans `experiment.yaml`.
`OVERWRITE=true` dans `prepare.sh` reconstruit les données normalisées du même split.

Pour une inférence qualitative ultérieure, utiliser le nouveau checkpoint avec
son `eval_reports/<run_name>/config_used.yaml` : ce YAML pointe sur les
statistiques de normalisation du train ARG_FRA. Les prédictions avec GT sur les
9 clips ARG_FRA sont déjà produites par l'évaluation automatique.
