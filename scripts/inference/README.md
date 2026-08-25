# Inférence qualitative sans GT

Le pipeline accepte les modèles `mlp`, `tcn` et `transformer`. Il faut fournir
le YAML exact associé au checkpoint, car les fichiers `.pt` ne contiennent ni
l'architecture ni la liste des features. Préférer :

```text
outputs/eval_reports/<run_name>/config_used.yaml
```

## Lancement Slurm

```bash
sbatch scripts/inference/run_inference.sh \
  transformer \
  outputs/eval_reports/root_transformer_v1_delta_new_root_init_wo_vj_gi_pp/config_used.yaml \
  best \
  --sequence ekstraklasa_001999
```

Les trois premiers arguments sont :

1. type de modèle : `mlp`, `tcn` ou `transformer` ;
2. configuration YAML exacte ;
3. checkpoint optionnel : `best`, `last` ou chemin vers un `.pt` (défaut : `best`).

Tous les arguments suivants sont transmis à `python -m field_converter.inference`.
Par exemple, pour forcer l'absence de distorsion et la taille de l'image :

```bash
sbatch scripts/inference/run_inference.sh transformer CONFIG best \
  --k-to-zero \
  --image-size 1920 1080
```

## Rééchantillonnage temporel optionnel

Le rééchantillonnage est activé uniquement si le FPS actuel des tableaux
d'entrée est fourni avec `--source-fps`. La cible vaut 50 FPS par défaut :

```bash
sbatch scripts/inference/run_inference.sh transformer CONFIG best \
  --sequence ekstraklasa_001999 \
  --source-fps 25
```

Avec le lancement Ekstraklasa codé comme valeur par défaut dans le script, la
forme courte équivalente est :

```bash
sbatch scripts/inference/run_inference.sh --source-fps 25
```

- si le FPS source est inférieur à 50, les boxes, squelettes, intrinsèques,
  translations et distorsions utilisent une interpolation cubique locale qui
  préserve la forme ; les rotations caméra utilisent une SLERP ;
- si le FPS source est supérieur à 50, la frame source temporellement la plus
  proche est sélectionnée ;
- à 50 FPS, les tableaux restent inchangés ;
- sans `--source-fps`, le rééchantillonnage reste désactivé.

Le pipeline ne crée pas de nouvelles images JPG. `predictions.npz` conserve
donc `source_frame_indices`, `source_frame_positions` (fractionnaires),
`timestamps_s`, `source_fps` et `output_fps` pour synchroniser une
visualisation avec les images source. La durée n'est jamais extrapolée : par
exemple, 1000 frames indexées de 0 à 999 à 25 FPS donnent 1999 timestamps à
50 FPS, de 0 à 39,96 s.

Sans `--sequence`, toutes les séquences complètes de `data/data_inference`
sont traitées.

### Stockage temporaire Slurm

Le script n'utilise volontairement ni `SLURM_TMPDIR` ni `/tmp`, car certains
nœuds peuvent renvoyer `Input/output error` sur leur disque temporaire local.
Chaque job utilise à la place :

```text
.cache/inference_jobs/<SLURM_JOB_ID>/
```

`TMPDIR`, les caches Matplotlib/XDG/Torch/Numba et les temporaires joblib sont
tous redirigés vers ce dossier. Les avertissements initiaux de `slurmstepd` sur
`/tmp/skhan` peuvent encore apparaître avant le démarrage du script, mais le
processus d'inférence n'utilise plus ce chemin ensuite.

## Données attendues

```text
data/data_inference/
  boxes/<sequence>.npy
  cameras/<sequence>.npz
  skel_2d/<sequence>.npy
  skel_3d_relative/<sequence>.npy
  frames/<sequence>/*.jpg       # optionnel, utilisé pour conserver les numéros de frames
```

Les tableaux peuvent être en `(T,N,...)` ou `(N,T,...)`. Les caméras doivent
contenir `K`, `R`, `t`; `k` est optionnel et vaut zéro s'il est absent. Le plan
du terrain vient par défaut de `data/pitch_points.txt` et définit le repère
canonique FIFA vers lequel les extrinsèques sont éventuellement alignées.

Le signe global des squelettes SAM3D doit aussi être le même qu'à
l'entraînement (`--sam3d-sign 1` par défaut). Les coordonnées monde de la
caméra sont reconstruites par `C = -R.T @ t`. Le mode
`--world-alignment auto`, actif par défaut, détecte une caméra placée de l'autre
côté du terrain ou des signes d'axes opposés et teste les rotations de 180° qui
conservent un repère droit. Utiliser `--world-alignment none` pour le désactiver.
Le résumé conserve la décision et signale les camera centers qui restent loin
de la distribution train après alignement.

Les hypothèses géométriques, les dimensions FIFA recommandées et la fenêtre de
positions caméra observée dans le train sont détaillées dans
`src/field_converter/inference/README.md`.

## Sorties

Les artefacts de préparation sont enregistrés dans :

```text
data/data_inference/features/
data/data_inference/features_normalized/
data/data_inference/ground_intersection/
data/data_inference/root_init_cam/
data/data_inference/root_init_cam_normalized/
```

Les prédictions finales utilisent un sous-dossier dédié, séparé des fichiers
d'évaluation avec GT :

```text
outputs/predictions/inference/<run_name>/<sequence>/
  predictions.npz
  root_predictions.csv
  summary.json
```

`predictions.npz` contient notamment `root_pred_m` (repère caméra),
`root_world_pred_m`, les joints 3D reconstruits, leurs projections 2D, les
roots initiaux, `camera_center_world_m`, les paramètres `K/R/t/k`, les
intersections sol, la chronologie source/cible et le nombre de fenêtres ayant
contribué à chaque position.
Les tableaux sont denses en `(N,T,...)` et les positions non prédites valent
`NaN`.

Quand un alignement est appliqué, `R`, les features et les clés `*_world_m`
sont dans le repère aligné sur le train. Les clés contenant `source_world` et
`R_source/t_source` permettent de revenir au repère original du dataset.

`summary.json` ne rapporte aucune métrique GT. L'erreur de reprojection contre
SAM2D est seulement un proxy de cohérence, pas une mesure de précision 3D.

Le YAML et le checkpoint doivent provenir du même run. Les checkpoints ne
conservent pas le schéma des entrées. De même, `--k-to-zero` doit correspondre
au prétraitement utilisé pour entraîner le modèle ; l'activer uniquement à
l'inférence change la distribution des features caméra.
