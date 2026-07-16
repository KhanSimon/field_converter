# V1 — Root translation refinement (frame-wise MLP)

## Objectif
Implémentation d’une baseline **propre, reproductible et minimaliste** pour affiner la translation du root en caméra.

## Contraintes V1 (volontairement simples)
- Modèle: **MLP frame-wise** (pas de temporel)
- Entrée: concaténation de features **par frame** (3D SAM relatives, bbox feat, cam feat, valid_joints…)
- Sortie: uniquement `root_cam_norm_pred` de forme `(B,3)` en **espace normalisé**
- Entraînement: régression `SmoothL1(root_pred_norm, root_gt_norm)` (+ pertes optionnelles désactivées par défaut)
- Évaluation: reconstruction 3D caméra + reprojection 2D pour métriques/plots

## Quickstart
Pré-requis: exécuter dans un environnement Python avec **PyTorch** (et NumPy, PyYAML, matplotlib, tqdm).

Train (config debug par défaut):
```bash
PYTHONPATH=src python -m field_converter.training.train_root_mlp --config configs/mlp/root_mlp_v1.yaml
```

Évaluer (métriques + prédictions + plots):
```bash
PYTHONPATH=src python -m field_converter.training.evaluate_root_mlp --config configs/mlp/root_mlp_v1.yaml --checkpoint best
```

Baseline + comparaison (mean-root):
```bash
PYTHONPATH=src python -m field_converter.training.compare_baseline_root --config configs/mlp/root_mlp_v1.yaml
```


# V1 — Root translation refinement (temporal TCN)

## Objectif
Version temporelle (fenêtres overlapées + agrégation par frame) qui prédit `root_cam_norm_pred` de forme `(B,T,3)`.

## Quickstart
Train:
```bash
PYTHONPATH=src python -m field_converter.training.train_root_tcn --config configs/tcn/root_tcn_v1.yaml
```

Évaluer (métriques + prédictions + plots):
```bash
PYTHONPATH=src python -m field_converter.training.evaluate_root_tcn --config configs/tcn/root_tcn_v1.yaml --checkpoint best
```

Comparer Baseline vs MLP vs TCN (barplot):
```bash
PYTHONPATH=src python -m field_converter.training.compare_root_models \
  --mlp_config configs/mlp/root_mlp_v1_train.yaml \
  --tcn_config configs/tcn/root_tcn_v1.yaml \
  --split valid
```


## Structure du code (ce qui a été ajouté)

### Config
- `configs/mlp/root_mlp_v1_train.yaml`: config V1 (train)
- `configs/mlp/root_mlp_v1_debug.yaml`: config V1 (debug)
- `src/field_converter/training/config.py`:
  - dataclasses typées: `RunConfig`, `InputConfig`, `DatasetConfig`, `ModelConfig`, `OptimizerConfig`, `TrainingConfig`, `LossWeights`, `EvalConfig`, `PlotsConfig`
  - loader `load_run_config(path)` (YAML/JSON)
  - `data_dir: auto` -> `field_converter.pathseeker.DATA_DIR / "features_normalized"`
  - chemins dérivés:
    - `outputs/checkpoints/<run_name>/...`
    - `outputs/predictions/<run_name>/...`
    - `outputs/eval_reports/<run_name>/...`

### Dataset
- `src/field_converter/training/dataset.py`:
  - `NormalizedFrameDataset`: Dataset PyTorch **frame-wise**
  - indexation de tous les `(person_idx, frame_idx)` où `valid_mask=True`
  - cache simple par séquence (évite de recharger le `.npz` à chaque sample)
  - optimisation mémoire: le cache ne charge plus tout le `.npz` mais uniquement les clés nécessaires selon `InputConfig`
  - `infer_input_dim(cfg)` pour dimensionner le MLP

### Performance (DataLoader / BeeGFS)
Sur des fichiers `.npz` par séquence (souvent compressés) et un filesystem réseau (ex: BeeGFS), un `shuffle=True` naïf peut provoquer beaucoup d'accès aléatoires et de décompression, ce qui ralentit fortement l'entraînement (GPU idle) et peut augmenter la RAM par worker. Pour limiter ça:
- le `DataLoader` utilise un `prefetch_factor` réduit quand `num_workers>0`
- option `training.group_batches_by_sequence: true`: les batches sont groupés par séquence (sampler dédié) pour maximiser la localité et profiter du cache par séquence
- le calcul de pertes optionnelles est court-circuité quand leurs poids sont à 0 (ex: `cam3d=0`, `proj=0`)

### Modèle
- `src/field_converter/models/mlp.py`:
  - MLP générique: séquence `Linear -> Activation -> Dropout` répétée, puis `Linear` de sortie
  - activations supportées: `relu`, `gelu`
- `src/field_converter/models/root_refiner.py`:
  - `RootRefiner`: wrapper `nn.Module` autour du MLP, sortie `(B,3)`

### Normalisation / dénormalisation
- `src/field_converter/utils/normalization.py`:
  - `TorchNormalizationStats.load(data_dir/normalization_stats.npz)`
  - helpers:
    - `denorm_root(root_norm) = root_norm * std_root + mean_root`
    - `denorm_sam3d_rel(X_norm) = X_norm * std_sam3d_rel + mean_sam3d_rel`

### Utils
- `src/field_converter/utils/torch_utils.py`:
  - `seed_everything(seed)` (python/numpy/torch)
  - `get_device(auto|cpu|cuda)`
- `src/field_converter/utils/io.py`:
  - `ensure_dir(path)`
  - `write_json(path, payload)`

### Géométrie
- `src/field_converter/geometry/transforms.py`:
  - conventions **row-vector**:
    - `X_cam = X_world @ R.T + t`
    - `X_world = (X_cam - t) @ R`
- `src/field_converter/geometry/projection.py`:
  - projection caméra -> image avec distorsion radiale `(k1,k2)`:
    - `x = X/Z`, `y = Y/Z`, `r2=x^2+y^2`, `factor=1+k1*r2+k2*r2^2`
    - `u=fx*x_d+cx`, `v=fy*y_d+cy`

### Losses
- `src/field_converter/losses/root_losses.py`:
  - `loss_root_smooth_l1(root_pred_norm, root_gt_norm)` (V1)
  - `reconstruct_X_cam_pred(x3d_sam_norm, root_pred_norm, stats)`:
    - dénormalise `X_rel` et `root`, puis `X_cam_pred = X_rel + root[:,None,:]`
  - `loss_cam3d(...)` optionnelle (masquée par `valid_joints`)
- `src/field_converter/losses/projection_losses.py`:
  - `loss_reprojection(...)` optionnelle en pixels (masquée)
- `src/field_converter/losses/temporal_losses.py`:
  - pertes temporelles masquées (utilisées en TCN)

#### $L_{root}$

Perte Smooth L1 entre $root\_pred\_norm$ et $root\_gt\_norm$ (réduction moyenne).

La fonction Smooth L1 (Huber variant) pour un résidu scalaire :

$$
r = pred - gt
$$

$$
\ell_{smoothL1}(r) =
\begin{cases}
0.5r^2 & \text{si } |r| < 1 \\
|r| - 0.5 & \text{sinon}
\end{cases}
$$

La perte globale est :

$$
L_{root} = \operatorname{mean}_{batch}\Big(\ell_{smoothL1}(\text{each coord})\Big)
$$

#### $L_{cam3d}$

Perte 3D caméra entre $X_{cam}^{pred}$ et $Y_{cam}^{gt}$.

On reconstruit d'abord :

$$
X_{cam}^{pred} =
\operatorname{denorm\_sam3d}(x3d\_sam\_norm)
+
\operatorname{denorm\_root}(root\_pred\_norm)
$$

Puis on applique une perte Smooth L1 (ou L1) par coordonnée,
moyennée par joint, avec réduction masquée :

$$
L_{cam3d}=
\frac{
\sum_{b,j} mask_{b,j}
\; \ell_{joint}
\left(
X_{b,j}^{pred},
Y_{b,j}^{gt}
\right)
}{
\sum_{b,j} mask_{b,j}
}
$$

#### $L_{proj}$

Même principe pour la 2d reprojetée en pixels. 

### Métriques / évaluation
- `src/field_converter/evaluation/metrics.py`:
  - `MetricsAccumulator`: accumule des métriques en unités dénormalisées
  - métriques produites:
    - `root_error_mean_m`, `root_error_median_m`, `root_error_p90_m`
    - `root_error_x_m`, `root_error_y_m`, `root_error_z_m` (erreur abs moyenne par axe)
    - `MPJPE_cam_m`, `MPJPE_world_m` (masqué `valid_joints`)
    - `reprojection_error_mean_px`, `reprojection_error_median_px` (masqué)
- `src/field_converter/evaluation/evaluator.py`:
  - `Evaluator.evaluate_split(model, dataloader, out_dir, split_name)`
  - sauvegarde optionnelle:
    - `outputs/predictions/<run_name>/<split>_predictions.npz`
    - `outputs/predictions/<run_name>/<split>_predictions.csv` (désactivé par défaut)
  - si `model` est un `nn.Module`: `.to(device)` + `.eval()` automatiquement

### Baseline
- `src/field_converter/evaluation/baseline.py`:
  - baseline A: `MeanRootBaseline(mean_root_norm)` (constante en normalisé)
  - `compute_mean_root_norm(train_dl, device)` calcule la moyenne du `root_gt` normalisé

### Plots
- `src/field_converter/evaluation/visualization.py` (backend matplotlib `Agg`):
  - `training_curves.png`: losses train/valid + root error valid
  - `root_xyz_timeseries_<split>.png`: GT vs Pred des 3 composantes root
  - `world_traj_xy_<split>.png`: trajectoire monde XY (root)
  - `reproj_overlay_<split>.png`: overlay 2D (GT 2D vs SAM 2D vs reprojection prédite)
  - `model_vs_baseline_<split>.png`: barplot comparatif (root error / MPJPE)

### Entraînement
- `src/field_converter/training/trainer.py`:
  - loop AdamW + gradient clipping optionnel
  - early stopping sur `root_error_mean_m` en validation
  - checkpoints:
    - `best.pt` (meilleur `root_error_mean_m`)
    - `last.pt`
  - log CSV: `outputs/eval_reports/<run_name>/train_log.csv`

### Entrypoints
Les entrypoints “officiels” sont des modules (plus robustes car `scripts/` est souvent ignoré côté git):
- `python -m field_converter.training.train_root_mlp`
- `python -m field_converter.training.evaluate_root_mlp`
- `python -m field_converter.training.train_root_tcn`
- `python -m field_converter.training.evaluate_root_tcn`
- `python -m field_converter.training.compare_root_models`
- `python -m field_converter.training.compare_baseline_root`

Ils correspondent à:
- `src/field_converter/training/train_root_mlp.py`
- `src/field_converter/training/evaluate_root_mlp.py`
- `src/field_converter/training/compare_baseline_root.py`

Des wrappers existent aussi dans `scripts/` (mêmes arguments) et délèguent aux modules.


## Format des données attendues (features_normalized)

### Chemin
Par défaut, `data_dir: auto` pointe vers:

```text
data/features_normalized/
  normalization_stats.npz
  split.json
  train/<sequence>.npz
  valid/<sequence>.npz
  test/<sequence>.npz
```

Pour le mode delta, `root_init_dir: auto` pointe vers:

```text
data/root_init_cam_normalized/
  split.json
  train/<sequence>.npy
  valid/<sequence>.npy
  test/<sequence>.npy
```

Ces fichiers se génèrent en deux étapes:

```bash
PYTHONPATH=src python -m field_converter.data_preparation.generate_root_init --features-dirname features --overwrite
PYTHONPATH=src python -m field_converter.data_preparation.normalize_root_init --features-normalized-dirname features_normalized --overwrite
```

### Fichiers `.npz` par séquence
Le dataset V1 lit (au minimum) les clés suivantes, avec des shapes typiques:

- `valid_mask`: `(P,T)` bool (indique quelles frames sont valides)
- `valid_joints`: `(P,T,25)` bool
- `skel_3d_sam3dbody_from_bbox_gt`: `(P,T,25,3)` **normalisé** (joints relatifs)
- `Y_root_cam_gt`: `(P,T,3)` **normalisé** (root caméra GT)
- `pitch_points_2d`: `(T,50,2)` **normalisé** par `(W,H)`; les points hors image valent zéro
- `valid_pitch_points`: `(T,50)` bool, masque des points terrain visibles

- `K`: `(T,3,3)` float32
- `R`: `(T,3,3)` float32
- `t`: `(T,3)` float32
- `k`: `(T,2)` float32 (distorsion radiale)

- `Y_cam_gt`: `(P,T,25,3)` float32 (mètres)
- `Y_2d_gt`: `(P,T,25,2)` float32 (pixels)

Les features terrain se régénèrent avec le pipeline habituel :

```bash
sbatch scripts/feature_engi/feature_creation.sh
# Une fois le job termine :
sbatch scripts/feature_engi/normalize.sh
```

`input_config.use_pitch_points_2d: true` ajoute les 100 coordonnées des 50
repères ainsi que leur masque de visibilité (50 valeurs), soit 150 dimensions
par frame. La même entrée est utilisée par le MLP, le TCN et le Transformer.

Entrées optionnelles supportées par `InputConfig` (si présentes dans le `.npz`):

- `skel_2d_sam3dbody_from_bbox_gt`: `(P,T,25,2)` (2D normalisé image)
- `skel_2d_sam3dbody_from_bbox_gt_box`: `(P,T,25,2)`
- `bbox_feat` ou `bbox_feat_clean`: `(P,T,5)`
- `cam_feat_*`: `(T,6)` pour `base_*` ou `(T,12)` pour `boosted_*`
- `ground_intersection`: `(P,T,3)` (point d'intersection sol normalisé)

Note: toute valeur NaN/Inf dans les features d’entrée est remplacée par 0 (les masks restent séparés).


## Configuration (YAML) — détails
Les fichiers dans config contrôlent tout le pipeline.

### Champs principaux
- `run_name`: nom du run (utilisé dans `outputs/.../<run_name>/...`)
- `seed`: seed pour numpy/torch
- `device`: `auto | cpu | cuda`
- `prediction_mode`: `absolute | delta`
  - `absolute`: le modèle sort directement `root_pred_norm`
  - `delta`: le modèle sort `delta_pred_norm`, puis `root_pred_norm = root_init_norm + delta_pred_norm`
- `data_dir`: `auto` ou chemin vers `data/features_normalized`
- `root_init_dir`: `auto` ou chemin vers `data/root_init_cam_normalized` (requis en mode `delta`)
- `output_dir`: par défaut `outputs/` à la racine du repo

### `input_config`
Active/désactive les composantes concaténées dans le vecteur `x` (frame-wise):
- `use_x3d_sam_rel` (25*3)
- `use_x2d_img` (25*2)
- `use_x2d_box` (25*2)
- `use_bbox_feat` (5)
- `bbox_clean_or_noisy`: `clean | noisy` (choix de clé `bbox_feat_clean` vs `bbox_feat`)
- `use_cam_feat` (6 ou 12)
- `cam_feat_type`: `base_clean | base_noisy | boosted_clean | boosted_noisy`
- `use_ground_intersection` (3)
- `use_valid_joints_as_input` (25)

### `dataset` (helpers debug)
- `max_sequences`: limite le nombre de séquences par split (ou `null`)
- `max_samples_per_sequence`: sous-échantillonne aléatoirement les frames valides d’une séquence (ou `null`)
- `subsample_stride`: stride déterministe sur les frames valides

### `loss_weights`
- `root`: poids de `SmoothL1` sur le root (actif)
- `cam3d`: poids de la loss 3D caméra (optionnelle)
- `proj`: poids de la loss reprojection (optionnelle)

Par défaut, V1 garde `cam3d=0` et `proj=0` (baseline propre “root-only”).


## Sorties générées

### Après entraînement
Dans `outputs/`:

- `checkpoints/<run_name>/best.pt`
- `checkpoints/<run_name>/last.pt`
- `eval_reports/<run_name>/train_log.csv`
- `eval_reports/<run_name>/train_summary.json`
- `eval_reports/<run_name>/config_used.yaml`

### Après évaluation
- `eval_reports/<run_name>/metrics.json`
- `predictions/<run_name>/<split>_predictions.npz`
  - contient: meta (seq/person/frame), root préd/gt en normalisé & mètres, root monde, erreur root, metrics JSON sérialisées
  - en mode `delta`, contient aussi `root_delta_pred_norm` et `root_init_norm`
- `eval_reports/<run_name>/plots/*.png` (si `plots.enabled: true`)

### Baseline
La baseline écrit dans:
- `outputs/eval_reports/<baseline_run_name>/metrics.json`
- `outputs/predictions/<baseline_run_name>/<split>_predictions.npz`
Et la comparaison est sauvée dans:
- `outputs/eval_reports/<run_name>/baseline_comparison.json`
- `outputs/eval_reports/<run_name>/plots/model_vs_baseline_<split>.png`


## Limitations connues (V1)
- Modèle strictement frame-wise (pas de cohérence temporelle)
- Les losses `cam3d` et `proj` sont disponibles mais désactivées par défaut
- Nécessite un env PyTorch fonctionnel (le code est prêt, mais les runs dépendent de l’environnement)


## Dépannage rapide
- Vérifier PyTorch:

```bash
python -c "import torch; print(torch.__version__)"
```

- Si `PYTHONPATH` est oublié: `ModuleNotFoundError: field_converter`.
  Utiliser exactement `PYTHONPATH=src` dans les commandes ci-dessus.
- Si les fichiers de data ne sont pas présents: `FileNotFoundError` sur `split.json` ou `train/<seq>.npz`.
  Vérifier `data_dir` dans la config.
