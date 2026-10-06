"""CPU checks using synthetic clips; no dataset, scheduler or GPU required."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from field_converter.ablation.common import PROJECT_ROOT, campaign_output_dir, fold_data_dirs
from field_converter.ablation.prepare import prepare_campaign
from field_converter.ablation.run_experiment import run_experiment
from field_converter.data_preparation.generate_root_init import ROOT_INIT_GENERATION_VERSION
from field_converter.training.tcn.config import load_tcn_run_config
from field_converter.training.transformer.config import load_transformer_run_config


spec = importlib.util.spec_from_file_location("arg_fra_workflow", Path(__file__).with_name("run.py"))
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class WorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        data = root / "source"
        features = data / "features"
        features.mkdir(parents=True)
        raw_roots = data / "root_init_cam"
        raw_roots.mkdir()
        values = {"ARG_CRO_001": 1.0, "ENG_FRA_001": 3.0, "BRA_KOR_001": 100.0, "ARG_FRA_001": -100.0}
        (data / "sequences_gt.txt").write_text("\n".join(values) + "\n")
        for sequence, value in values.items():
            roots = np.full((1, 3, 3), value, dtype=np.float32)
            rng = np.random.default_rng(7)
            np.savez_compressed(
                features / f"{sequence}.npz",
                K=np.tile(np.eye(3, dtype=np.float32), (3, 1, 1)),
                valid_mask=np.ones((1, 3), dtype=bool),
                valid_joints=np.ones((1, 3, 25), dtype=bool),
                skel_3d_sam3dbody_from_bbox_gt=rng.normal(size=(1, 3, 25, 3)).astype(np.float32),
                skel_2d_sam3dbody_from_bbox_gt=np.full((1, 3, 25, 2), [500.0, 400.0], dtype=np.float32),
                Y_rel_cam_gt=rng.normal(size=(1, 3, 25, 3)).astype(np.float32),
                Y_root_cam_gt=roots,
                ground_intersection=roots,
                cam_feat_boosted_clean=np.zeros((3, 12), dtype=np.float32),
                image_size=np.array([1920, 1080]),
                boxes_xyxy=np.tile(np.array([400.0, 300.0, 600.0, 700.0]), (1, 3, 1)),
            )
            np.save(raw_roots / f"{sequence}.npy", roots)
        (raw_roots / "root_init_generation_meta.json").write_text(json.dumps({
            "generation_version": ROOT_INIT_GENERATION_VERSION, "pelvis_mode": "hips_mean",
        }))
        self.manifest = yaml.safe_load(workflow.DEFAULT_MANIFEST.read_text())
        self.manifest.update(data_dir=str(data), output_root=str(root / "outputs"), data_output_root=str(root / "data"))
        self.manifest["base_configs"] = {
            key: str(PROJECT_ROOT / value) for key, value in self.manifest["base_configs"].items()
        }
        self.path = root / "experiment.yaml"
        self.path.write_text(yaml.safe_dump(self.manifest, sort_keys=False))

    def test_paper_configs_and_whole_match_split(self) -> None:
        plan = workflow.generate_plan(self.path)
        self.assertEqual(plan["folds"]["arg_fra"]["counts"], {"train": 2, "valid": 1, "test": 1})
        self.assertEqual(plan["folds"]["arg_fra"]["sequences"]["test"], ["ARG_FRA_001"])
        self.assertEqual([run["seed"] for run in plan["runs"]], [2027, 1235])
        for run, loader in zip(plan["runs"], (load_tcn_run_config, load_transformer_run_config)):
            cfg = yaml.safe_load(Path(run["config_path"]).read_text())
            source = yaml.safe_load(Path(self.manifest["base_configs"][run["architecture"]]).read_text())
            for key in ("run_name", "data_dir", "root_init_dir", "output_dir"):
                cfg.pop(key)
                source.pop(key)
            self.assertEqual(cfg, source)
            self.assertEqual(loader(run["config_path"]).dataset.window_size, 41)

    def test_normalization_uses_new_train_and_resume_keeps_files(self) -> None:
        workflow.generate_plan(self.path)
        with self.assertRaisesRegex(RuntimeError, "Preprocessing is missing"):
            workflow.check_prepared(self.path)
        prepare_campaign(self.path)
        workflow.check_prepared(self.path)
        features, roots = fold_data_dirs(self.manifest, "arg_fra")
        stats_path = features / "normalization_stats.json"
        stats = json.loads(stats_path.read_text())
        np.testing.assert_allclose(stats["mean_root"], [2.0, 2.0, 2.0])
        np.testing.assert_allclose(np.load(roots / "test/ARG_FRA_001.npy"), -102.0)
        old_mtime = (features / "train/ENG_FRA_001.npz").stat().st_mtime_ns
        prepare_campaign(self.path)
        self.assertEqual(old_mtime, (features / "train/ENG_FRA_001.npz").stat().st_mtime_ns)
        stats["train_sequences"].append("ARG_FRA_001")
        stats_path.write_text(json.dumps(stats))
        with self.assertRaisesRegex(RuntimeError, "Normalization must use only"):
            workflow.check_prepared(self.path)

    def test_changed_split_cannot_replace_frozen_plan(self) -> None:
        workflow.generate_plan(self.path)
        plan_path = campaign_output_dir(self.manifest) / "plan.json"
        original = plan_path.read_bytes()
        self.manifest["folds"]["arg_fra"]["valid_match"] = "ENG_FRA"
        self.path.write_text(yaml.safe_dump(self.manifest))
        with self.assertRaisesRegex(RuntimeError, "new campaign_name"):
            workflow.generate_plan(self.path)
        self.assertEqual(plan_path.read_bytes(), original)

    def test_submission_dependency_and_dry_run(self) -> None:
        with patch.object(workflow.subprocess, "run", side_effect=[
            subprocess.CompletedProcess([], 0, "110;cluster\n", ""),
            subprocess.CompletedProcess([], 0, "111\n", ""),
        ]) as sbatch:
            result = workflow.submit(self.path, dry_run=False)
            self.assertEqual(result["jobs"], {"preprocessing": "110", "training_array": "111"})
            self.assertIn("--dependency=afterok:110", sbatch.call_args_list[1].args[0])
        submission = campaign_output_dir(self.manifest) / "submission.json"
        original = submission.read_bytes()
        with patch.object(workflow.subprocess, "run") as sbatch:
            workflow.submit(self.path, dry_run=True)
            sbatch.assert_not_called()
        self.assertEqual(submission.read_bytes(), original)

    def test_train_then_evaluate_and_resume(self) -> None:
        plan = workflow.generate_plan(self.path)
        with patch("field_converter.ablation.run_experiment.subprocess.run") as child:
            run_experiment(self.path, 1)
            commands = [call.args[0] for call in child.call_args_list]
            self.assertEqual(commands[0][2], "field_converter.training.train_root_transformer")
            self.assertEqual(commands[1][2], "field_converter.training.evaluate_root_transformer")
            self.assertIn("best", commands[1])
            self.assertIn("--no_baseline", commands[1])
        run = plan["runs"][1]
        output = campaign_output_dir(self.manifest)
        checkpoint = output / "checkpoints" / run["run_name"] / "best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        report = output / "eval_reports" / run["run_name"]
        report.mkdir(parents=True)
        (report / "train_summary.json").write_text("{}")
        shutil.copyfile(run["config_path"], report / "config_used.yaml")
        with patch("field_converter.ablation.run_experiment.subprocess.run") as child:
            run_experiment(self.path, 1)
            self.assertEqual(child.call_count, 1)
            self.assertEqual(child.call_args.args[0][2], "field_converter.training.evaluate_root_transformer")
        (report / "metrics.json").write_text(json.dumps({
            "splits": {split: {"root_error_mean_m": 0.1} for split in ("valid", "test")},
        }))
        with patch("field_converter.ablation.run_experiment.subprocess.run") as child:
            self.assertTrue(run_experiment(self.path, 1)["skipped"])
            child.assert_not_called()


if __name__ == "__main__":
    unittest.main()
