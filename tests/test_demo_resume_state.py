from __future__ import annotations

import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


class DemoResumeStateTests(unittest.TestCase):
    @staticmethod
    def _rewrite_checkpoint_payload(checkpoint_path: Path, mutate) -> None:
        manifest = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        state_path = checkpoint_path.parent / manifest["current"]["file"]
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        mutate(payload)
        torch.save(payload, state_path)
        manifest["current"]["bytes"] = state_path.stat().st_size
        manifest["current"]["sha256"] = hashlib.sha256(
            state_path.read_bytes()
        ).hexdigest()
        checkpoint_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _identity() -> object:
        from tools.demo_resume_state import RunIdentity

        return RunIdentity(
            baseline_commit="b4f323a",
            config_sha256="1" * 64,
            clip_sha256="2" * 64,
            recovery_code_sha256="3" * 64,
            python_version="3.10.0",
            torch_version="2.5.1+cu121",
            cuda_version="12.1",
            device_name="test-device",
            runtime_sha256="4" * 64,
            parity_epoch=10,
            parity_reference_sha256="5" * 64,
        )

    def test_checkpoint_restores_training_objects_and_rng_exactly(self) -> None:
        from tools.demo_resume_state import (
            restore_training_checkpoint,
            save_training_checkpoint,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "latest.pt"
            model = torch.nn.Linear(2, 1)
            center_criterion = torch.nn.Linear(1, 1, bias=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            optimizer_center = torch.optim.SGD(
                center_criterion.parameters(), lr=0.5
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=0.1
            )
            scaler = torch.amp.GradScaler("cpu")

            with torch.no_grad():
                model.weight.copy_(torch.tensor([[2.0, -1.0]]))
                model.bias.copy_(torch.tensor([0.5]))
                center_criterion.weight.fill_(3.0)
            loss = model(torch.tensor([[1.0, 2.0]])).square().sum()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            random.seed(101)
            np.random.seed(202)
            torch.manual_seed(303)
            identity = self._identity()
            save_training_checkpoint(
                checkpoint_path,
                epoch=7,
                phase="post_train",
                best_index={
                    "mAP": 0.7,
                    "Rank-1": 0.8,
                    "Rank-5": 0.9,
                    "Rank-10": 1.0,
                },
                identity=identity,
                model=model,
                optimizer=optimizer,
                center_criterion=center_criterion,
                optimizer_center=optimizer_center,
                scheduler=scheduler,
                scaler=scaler,
            )

            with torch.no_grad():
                model.weight.zero_()
                model.bias.zero_()
                center_criterion.weight.zero_()
            optimizer.param_groups[0]["lr"] = 99.0
            random.random()
            np.random.rand()
            torch.rand(())

            cursor = restore_training_checkpoint(
                checkpoint_path,
                expected_identity=identity,
                model=model,
                optimizer=optimizer,
                center_criterion=center_criterion,
                optimizer_center=optimizer_center,
                scheduler=scheduler,
                scaler=scaler,
            )

            self.assertEqual((cursor.epoch, cursor.phase), (7, "post_train"))
            self.assertEqual(cursor.best_index["mAP"], 0.7)
            self.assertAlmostEqual(
                model(torch.tensor([[1.0, 2.0]])).item(), 0.46, places=3
            )
            self.assertAlmostEqual(center_criterion.weight.item(), 3.0)
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.001)
            self.assertTrue(optimizer.state_dict()["state"])
            self.assertAlmostEqual(random.random(), 0.5811521325045647)
            self.assertAlmostEqual(float(np.random.rand()), 0.22420617998962666)
            self.assertAlmostEqual(float(torch.rand(())), 0.03799790143966675)
            self.assertFalse(list(checkpoint_path.parent.glob("*.tmp-*")))

    def test_resume_cursor_selects_the_only_valid_next_action(self) -> None:
        from tools.demo_resume_state import (
            ResumeCursor,
            next_resume_action,
        )

        best_index = {
            "mAP": 0.7,
            "Rank-1": 0.8,
            "Rank-5": 0.9,
            "Rank-10": 1.0,
        }
        cases = [
            (ResumeCursor(0, "post_eval", best_index), 50, 1, ("train", 1)),
            (
                ResumeCursor(7, "post_train", best_index),
                50,
                1,
                ("evaluate", 7),
            ),
            (
                ResumeCursor(7, "post_train", best_index),
                50,
                10,
                ("finalize", 7),
            ),
            (ResumeCursor(7, "post_eval", best_index), 50, 1, ("train", 8)),
            (
                ResumeCursor(50, "post_eval", best_index),
                50,
                1,
                ("complete", 50),
            ),
        ]

        for cursor, max_epochs, eval_period, expected in cases:
            with self.subTest(cursor=cursor, eval_period=eval_period):
                action = next_resume_action(
                    cursor, max_epochs=max_epochs, eval_period=eval_period
                )
                self.assertEqual((action.kind, action.epoch), expected)

    def test_epoch_zero_post_train_is_rejected_at_the_checkpoint_boundary(self) -> None:
        from tools.demo_resume_state import save_training_checkpoint

        model = torch.nn.Linear(1, 1)
        center = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters())
        optimizer_center = torch.optim.SGD(center.parameters(), lr=0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        scaler = torch.amp.GradScaler("cpu")
        best_zero = {
            "mAP": 0.0,
            "Rank-1": 0.0,
            "Rank-5": 0.0,
            "Rank-10": 0.0,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "post_train.*epoch zero"):
                save_training_checkpoint(
                    Path(temporary_directory) / "latest.pt",
                    epoch=0,
                    phase="post_train",
                    best_index=best_zero,
                    identity=self._identity(),
                    model=model,
                    optimizer=optimizer,
                    center_criterion=center,
                    optimizer_center=optimizer_center,
                    scheduler=scheduler,
                    scaler=scaler,
                )

    def test_tampered_but_loadable_state_is_rejected_by_manifest_digest(self) -> None:
        from tools.demo_resume_state import (
            restore_training_checkpoint,
            save_training_checkpoint,
        )

        model = torch.nn.Linear(1, 1)
        center = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters())
        optimizer_center = torch.optim.SGD(center.parameters(), lr=0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        scaler = torch.amp.GradScaler("cpu")
        state_objects = {
            "model": model,
            "optimizer": optimizer,
            "center_criterion": center,
            "optimizer_center": optimizer_center,
            "scheduler": scheduler,
            "scaler": scaler,
        }
        best_zero = {
            "mAP": 0.0,
            "Rank-1": 0.0,
            "Rank-5": 0.0,
            "Rank-10": 0.0,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "latest.json"
            save_training_checkpoint(
                checkpoint_path,
                epoch=0,
                phase="post_eval",
                best_index=best_zero,
                identity=self._identity(),
                **state_objects,
            )
            manifest = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            state_path = checkpoint_path.parent / manifest["current"]["file"]
            payload = torch.load(state_path, map_location="cpu", weights_only=False)
            payload["model"]["weight"].add_(1.0)
            torch.save(payload, state_path)
            manifest["current"]["bytes"] = state_path.stat().st_size
            checkpoint_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                restore_training_checkpoint(
                    checkpoint_path,
                    expected_identity=self._identity(),
                    **state_objects,
                )

    def test_trained_epoch_rejects_an_empty_main_optimizer_state(self) -> None:
        from tools.demo_resume_state import save_training_checkpoint

        model = torch.nn.Linear(1, 1)
        center = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters())
        optimizer_center = torch.optim.SGD(center.parameters(), lr=0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        scaler = torch.amp.GradScaler("cpu")
        best_zero = {
            "mAP": 0.0,
            "Rank-1": 0.0,
            "Rank-5": 0.0,
            "Rank-10": 0.0,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "optimizer state.*empty"):
                save_training_checkpoint(
                    Path(temporary_directory) / "latest.json",
                    epoch=1,
                    phase="post_train",
                    best_index=best_zero,
                    identity=self._identity(),
                    model=model,
                    optimizer=optimizer,
                    center_criterion=center,
                    optimizer_center=optimizer_center,
                    scheduler=scheduler,
                    scaler=scaler,
                )

    def test_digest_valid_checkpoint_rejects_a_missing_scheduler_field(self) -> None:
        from tools.demo_resume_state import (
            restore_training_checkpoint,
            save_training_checkpoint,
        )

        model = torch.nn.Linear(1, 1)
        center = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters())
        optimizer_center = torch.optim.SGD(center.parameters(), lr=0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        scaler = torch.amp.GradScaler("cpu")
        optimizer.zero_grad()
        model(torch.tensor([[1.0]])).sum().backward()
        optimizer.step()
        scheduler.step()
        state_objects = {
            "model": model,
            "optimizer": optimizer,
            "center_criterion": center,
            "optimizer_center": optimizer_center,
            "scheduler": scheduler,
            "scaler": scaler,
        }
        best_zero = {
            "mAP": 0.0,
            "Rank-1": 0.0,
            "Rank-5": 0.0,
            "Rank-10": 0.0,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "latest.json"
            save_training_checkpoint(
                checkpoint_path,
                epoch=1,
                phase="post_train",
                best_index=best_zero,
                identity=self._identity(),
                **state_objects,
            )

            def delete_scheduler_field(payload) -> None:
                del payload["scheduler"][next(iter(payload["scheduler"]))]

            self._rewrite_checkpoint_payload(
                checkpoint_path, delete_scheduler_field
            )
            with self.assertRaisesRegex(ValueError, "scheduler state keys"):
                restore_training_checkpoint(
                    checkpoint_path,
                    expected_identity=self._identity(),
                    **state_objects,
                )

    def test_digest_valid_checkpoint_rejects_a_missing_adam_moment(self) -> None:
        from tools.demo_resume_state import (
            restore_training_checkpoint,
            save_training_checkpoint,
        )

        model = torch.nn.Linear(1, 1)
        center = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters())
        optimizer_center = torch.optim.SGD(center.parameters(), lr=0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        scaler = torch.amp.GradScaler("cpu")
        optimizer.zero_grad()
        model(torch.tensor([[1.0]])).sum().backward()
        optimizer.step()
        scheduler.step()
        state_objects = {
            "model": model,
            "optimizer": optimizer,
            "center_criterion": center,
            "optimizer_center": optimizer_center,
            "scheduler": scheduler,
            "scaler": scaler,
        }
        best_zero = {
            "mAP": 0.0,
            "Rank-1": 0.0,
            "Rank-5": 0.0,
            "Rank-10": 0.0,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "latest.json"
            save_training_checkpoint(
                checkpoint_path,
                epoch=1,
                phase="post_train",
                best_index=best_zero,
                identity=self._identity(),
                **state_objects,
            )

            def delete_adam_moment(payload) -> None:
                first_parameter_state = next(
                    iter(payload["optimizer"]["state"].values())
                )
                del first_parameter_state["exp_avg_sq"]

            self._rewrite_checkpoint_payload(
                checkpoint_path, delete_adam_moment
            )
            with self.assertRaisesRegex(ValueError, "Adam state keys"):
                restore_training_checkpoint(
                    checkpoint_path,
                    expected_identity=self._identity(),
                    **state_objects,
                )

    def test_digest_valid_checkpoint_rejects_duplicate_adam_parameter_ids(self) -> None:
        from tools.demo_resume_state import (
            restore_training_checkpoint,
            save_training_checkpoint,
        )

        model = torch.nn.Module()
        model.register_parameter(
            "left", torch.nn.Parameter(torch.tensor([1.0]))
        )
        model.register_parameter(
            "right", torch.nn.Parameter(torch.tensor([2.0]))
        )
        center = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.Adam(
            [{"params": [model.left]}, {"params": [model.right]}]
        )
        optimizer_center = torch.optim.SGD(center.parameters(), lr=0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        scaler = torch.amp.GradScaler("cpu")
        optimizer.zero_grad()
        (model.left.sum() + model.right.sum()).backward()
        optimizer.step()
        scheduler.step()
        state_objects = {
            "model": model,
            "optimizer": optimizer,
            "center_criterion": center,
            "optimizer_center": optimizer_center,
            "scheduler": scheduler,
            "scaler": scaler,
        }
        best_zero = {
            "mAP": 0.0,
            "Rank-1": 0.0,
            "Rank-5": 0.0,
            "Rank-10": 0.0,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "latest.json"
            save_training_checkpoint(
                checkpoint_path,
                epoch=1,
                phase="post_train",
                best_index=best_zero,
                identity=self._identity(),
                **state_objects,
            )

            def duplicate_parameter_id(payload) -> None:
                first_id = payload["optimizer"]["param_groups"][0]["params"][0]
                second_id = payload["optimizer"]["param_groups"][1]["params"][0]
                payload["optimizer"]["param_groups"][1]["params"][0] = first_id
                del payload["optimizer"]["state"][second_id]

            self._rewrite_checkpoint_payload(
                checkpoint_path, duplicate_parameter_id
            )
            with self.assertRaisesRegex(ValueError, "Adam parameter IDs"):
                restore_training_checkpoint(
                    checkpoint_path,
                    expected_identity=self._identity(),
                    **state_objects,
                )

    def test_evaluation_crash_resume_matches_uninterrupted_training(self) -> None:
        from tools.demo_resume_state import run_resumable_epochs

        best_zero = {
            "mAP": 0.0,
            "Rank-1": 0.0,
            "Rank-5": 0.0,
            "Rank-10": 0.0,
        }

        def seed_all() -> None:
            random.seed(17)
            np.random.seed(23)
            torch.manual_seed(29)

        def make_objects() -> dict[str, object]:
            model = torch.nn.Linear(1, 1)
            center = torch.nn.Linear(1, 1, bias=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
            optimizer_center = torch.optim.SGD(center.parameters(), lr=0.5)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=0.5
            )
            return {
                "model": model,
                "optimizer": optimizer,
                "center_criterion": center,
                "optimizer_center": optimizer_center,
                "scheduler": scheduler,
                "scaler": torch.amp.GradScaler("cpu"),
            }

        def make_train_epoch(objects: dict[str, object]):
            def train_epoch(_epoch: int) -> None:
                model = objects["model"]
                optimizer = objects["optimizer"]
                scheduler = objects["scheduler"]
                optimizer.zero_grad()
                value = (
                    random.random()
                    + float(np.random.rand())
                    + float(torch.rand(()))
                )
                loss = model(torch.tensor([[value]])).square().sum()
                loss.backward()
                optimizer.step()
                scheduler.step()

            return train_epoch

        def result(objects: dict[str, object]) -> tuple[float, float, float]:
            model = objects["model"]
            optimizer = objects["optimizer"]
            parameter = next(model.parameters())
            exp_avg = optimizer.state[parameter]["exp_avg"]
            return (
                float(parameter.detach().flatten()[0]),
                float(exp_avg.flatten()[0]),
                float(optimizer.param_groups[0]["lr"]),
            )

        def next_rng() -> tuple[float, float, float]:
            return (
                random.random(),
                float(np.random.rand()),
                float(torch.rand(())),
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            recovered_checkpoint = (
                Path(temporary_directory) / "recovered" / "latest.pt"
            )
            control_checkpoint = (
                Path(temporary_directory) / "control" / "latest.pt"
            )

            seed_all()
            interrupted_objects = make_objects()
            attempted_evaluations: list[int] = []

            def crashing_eval(epoch: int, best_index: dict[str, float]):
                attempted_evaluations.append(epoch)
                torch.rand(())
                raise RuntimeError("injected evaluation crash")

            with self.assertRaisesRegex(RuntimeError, "injected evaluation crash"):
                run_resumable_epochs(
                    checkpoint_path=recovered_checkpoint,
                    identity=self._identity(),
                    max_epochs=2,
                    eval_period=1,
                    initial_best_index=best_zero,
                    train_epoch=make_train_epoch(interrupted_objects),
                    evaluate_epoch=crashing_eval,
                    **interrupted_objects,
                )
            self.assertEqual(attempted_evaluations, [1])

            resumed_objects = make_objects()
            resumed_evaluations: list[int] = []

            def resumed_eval(epoch: int, best_index: dict[str, float]):
                resumed_evaluations.append(epoch)
                torch.rand(())
                updated = dict(best_index)
                updated["mAP"] = epoch / 10
                return updated

            recovered_cursor = run_resumable_epochs(
                checkpoint_path=recovered_checkpoint,
                identity=self._identity(),
                max_epochs=2,
                eval_period=1,
                initial_best_index=best_zero,
                train_epoch=make_train_epoch(resumed_objects),
                evaluate_epoch=resumed_eval,
                **resumed_objects,
            )
            recovered_result = result(resumed_objects)
            recovered_rng = next_rng()

            seed_all()
            control_objects = make_objects()
            control_evaluations: list[int] = []

            def control_eval(epoch: int, best_index: dict[str, float]):
                control_evaluations.append(epoch)
                torch.rand(())
                updated = dict(best_index)
                updated["mAP"] = epoch / 10
                return updated

            control_cursor = run_resumable_epochs(
                checkpoint_path=control_checkpoint,
                identity=self._identity(),
                max_epochs=2,
                eval_period=1,
                initial_best_index=best_zero,
                train_epoch=make_train_epoch(control_objects),
                evaluate_epoch=control_eval,
                **control_objects,
            )
            control_result = result(control_objects)
            control_rng = next_rng()

            self.assertEqual(resumed_evaluations, [1, 2])
            self.assertEqual(control_evaluations, [1, 2])
            self.assertEqual(
                (recovered_cursor.epoch, recovered_cursor.phase),
                (2, "post_eval"),
            )
            self.assertEqual(recovered_cursor, control_cursor)
            self.assertEqual(recovered_result, control_result)
            self.assertEqual(recovered_rng, control_rng)


if __name__ == "__main__":
    unittest.main()
