# Inférence qualitative sans vérité terrain

Ce package prépare un dataset externe, charge un modèle MLP/TCN/Transformer et
produit des prédictions denses de root sans utiliser de labels 3D.

## Repère et dimensions du terrain

Pour rester proche de la distribution d'entraînement, le terrain externe doit
idéalement suivre le repère FIFA utilisé par `data/pitch_points.txt` :

- dimensions proches de **105 m × 68 m** ;
- origine au centre du terrain ;
- longueur selon `x`, approximativement `[-52.5, 52.5]` m ;
- largeur selon `y`, approximativement `[-34, 34]` m ;
- plan du sol `z = 0`, avec `z` positif vers le haut dans le repère canonique.

Une différence de taille, d'origine ou d'échelle n'est pas corrigée par le
pipeline. Une homographie ou une calibration vers ce terrain canonique doit
être réalisée en amont si le dataset utilise un autre modèle de terrain.

## Position de caméra recommandée

Dans le training set actuel, les camera centers monde `C = -R.T @ t` sont dans
la fenêtre observée suivante :

| Axe | Minimum train | Maximum train | Moyenne ± écart-type |
|---|---:|---:|---:|
| `x` | -0.128 m | 0.323 m | 0.112 ± 0.120 m |
| `y` | -88.155 m | -66.729 m | -75.181 ± 6.200 m |
| `z` | 11.765 m | 19.039 m | 16.459 ± 2.102 m |

La caméra externe devrait idéalement se trouver dans, ou près de, cette
fenêtre. Une caméra réellement plus haute ou décalée reste hors distribution,
même après correction du côté du stade.

## Détection du côté opposé

`--world-alignment auto` est activé par défaut. Le pipeline compare la caméra
source à la distribution train avec quatre repères droits possibles :

- identité ;
- rotation de 180° autour de `x` : `(x,y,z) -> (x,-y,-z)` ;
- rotation de 180° autour de `y` : `(x,y,z) -> (-x,y,-z)` ;
- rotation de 180° autour de `z` : `(x,y,z) -> (-x,-y,z)`.

Une rotation automatique est appliquée seulement si :

1. la caméra source est à au moins 5 écarts-types du train sur un axe ;
2. la meilleure rotation réduit d'au moins 25 % le score RMS des z-scores.

Les fichiers de `data/data_inference` ne sont jamais modifiés. La rotation est
prise en compte pour les extrinsèques générées, le camera center, la direction
de caméra, les pitch points projetés, les rayons, les intersections sol, les
features caméra, les features normalisées et les sorties monde.

La translation extrinsèque `t`, les intrinsèques `K`, la distorsion `k`, les
pixels SAM2D, les boxes et les squelettes relatifs caméra restent inchangés.
Les prédictions exportent à la fois le repère aligné et le repère source.

Pour désactiver ou forcer la décision :

```bash
--world-alignment none
--world-alignment rotate_x_180
--world-alignment rotate_y_180
--world-alignment rotate_z_180
```

La correction ne traite volontairement que les inversions d'axes autour de
l'origine du terrain. Elle ne corrige ni une mauvaise unité, ni une translation
du terrain, ni une calibration `R/t` erronée.

## Entrées

```text
data/data_inference/
  boxes/<sequence>.npy
  cameras/<sequence>.npz       # K, R, t et éventuellement k
  skel_2d/<sequence>.npy
  skel_3d_relative/<sequence>.npy
  frames/<sequence>/*.jpg      # optionnel
```

Les tableaux personnes/temps peuvent être en `(T,N,...)` ou `(N,T,...)`. Le
fichier `pitch_points.txt` doit décrire le terrain canonique et les dimensions
d'image sont lues depuis `K`, ou fournies avec `--image-size WIDTH HEIGHT`.

## Artefacts

Le preprocessing écrit par défaut :

```text
data/data_inference/features/
data/data_inference/features_normalized/
data/data_inference/ground_intersection/
data/data_inference/root_init_cam/
data/data_inference/root_init_cam_normalized/
```

Les `.npz` conservent notamment `R_source`, `t_source`, la rotation choisie et
les camera centers avant/après alignement. Les prédictions finales vont dans :

```text
outputs/predictions/inference/<run_name>/<sequence>/
```

Le YAML exact du run est obligatoire : un checkpoint `.pt` ne contient pas à
lui seul l'architecture et la sélection des features.
