# field_converter.play_vizu

Visualisation minimale d'une "action" (un triplet `seq_name`, `frame_idx`, `player_idx`).

Le script sauvegarde dans `play_vizu/` une image de la frame correspondante avec:
- la bounding box GT du joueur
- optionnellement les keypoints 2D **SAM3DBody** (`skel_2d_sam3dbody_from_bbox_gt`)
- optionnellement les keypoints 2D **GT** (`Y_2d_gt`)

Pendant l'exécution, le script affiche aussi dans le terminal un résumé des données associées
(validité, bbox dérivée, et paramètres caméra GT au frame).

## Usage

Depuis la racine du repo:

```bash
PYTHONPATH=src python -m field_converter.play_vizu.viz_action \
  --seq_name NET_ARG_001438 \
  --frame_idx 100 \
  --player_idx 21 \
  --show_sam2d \
  --show_gt2d
```

Sortie:
- `play_vizu/ARG_CRO_220001_f00123_p003.jpg`

## Notes

- `frame_idx` et `player_idx` sont supposés **0-based** (comme dans `test_prediction.csv`).
- Le titre de la figure indique `valid=True/False` selon `data/valid_mask/<seq>.npy`.
- Les images sont lues depuis `data/images_gt/<seq_name>/<frame:05d>.jpg`.
- Les arrays sont lus depuis `data/boxes_gt`, `data/Y_2d_gt`, `data/skel_2d_sam3dbody_from_bbox_gt`, `data/valid_mask`, `data/valid_joints`.
- Certains fichiers sont en `(T,N,...)` et d'autres en `(N,T,...)` : le script s'aligne sur `valid_mask`.
