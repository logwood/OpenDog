# encoding: utf-8
"""Regression checks for the persistent latent-workspace experiment."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.modeling.backbones.resnest import Bottleneck, ResNeSt
from fastreid.modeling.meta_arch.latent_workspace import (
    PersistentLatentWorkspace,
    _CompetitiveMeshRead,
    _IdentityQueryHead,
    _SlotSetHead,
    _orthogonal_role_anchors,
    _sincos_2d_position,
    _sinkhorn_log_domain,
)
from fastreid.solver.build import get_default_optimizer_params
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.events import EventStorage
from pet_id import add_retri_config
from pet_id.latent_hooks import LatentHealthHook
from pet_id.workspace_paths import WORKSPACE_ROOT


CONFIG_ROOT = WORKSPACE_ROOT / "src" / "Pet-ReID-IMAG" / "configs"


def _load_tool_module(name):
    path = Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load test tool module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


microfit_report = _load_tool_module("analyze_latent_microfit")._report
_load_workspace_weights = _load_tool_module(
    "preflight_role_anchored_latent"
)._load_workspace_weights


class LatentWorkspaceTest(unittest.TestCase):
    @staticmethod
    def _workspace(dim=64, read_mode="mha"):
        return PersistentLatentWorkspace(
            dim=dim,
            num_slots=8,
            num_heads=8,
            mlp_ratio=4.0,
            gate_min=0.05,
            gate_init=0.10,
            dropout=0.0,
            read_mode=read_mode,
        )

    def test_rectangular_sinkhorn_matches_both_marginals(self):
        torch.manual_seed(3)
        cost = torch.randn(2, 11, 4)
        token_marginal = torch.full((2, 11), 1.0 / 11)
        slot_marginal = torch.full((2, 4), 1.0 / 4)
        transport, _, _ = _sinkhorn_log_domain(
            cost,
            token_marginal,
            slot_marginal,
            iterations=100,
            temperature=1.0,
        )
        torch.testing.assert_close(
            transport.sum(dim=2), token_marginal, rtol=1e-4, atol=1e-5
        )
        torch.testing.assert_close(
            transport.sum(dim=1), slot_marginal, rtol=1e-4, atol=1e-5
        )

    def test_mesh_breaks_identical_slot_ties_deterministically_in_eval(self):
        torch.manual_seed(5)
        mesh_read = _CompetitiveMeshRead(
            dim=16,
            num_slots=4,
            mlp_ratio=2.0,
            mesh_iterations=4,
            sinkhorn_iterations=5,
            mesh_lr=1.0,
            noise_std=1e-3,
            temperature=1.0,
        ).eval()
        identical_slots = torch.zeros(2, 4, 16)
        context = torch.randn(2, 20, 16)

        first, diagnostics, first_attention = mesh_read(
            identical_slots, context, collect_diagnostics=True
        )
        second, _, second_attention = mesh_read(
            identical_slots, context, collect_diagnostics=True
        )

        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(first_attention, second_attention, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(first).all())
        self.assertLess(diagnostics["attention_cosine"].item(), 0.999)
        self.assertGreater(diagnostics["effective_slots"].item(), 3.9)

    def test_mesh_uses_uniform_slot_capacity(self):
        mesh_read = _CompetitiveMeshRead(
            dim=16,
            num_slots=4,
            mlp_ratio=2.0,
            mesh_iterations=2,
            sinkhorn_iterations=5,
            mesh_lr=1.0,
            noise_std=1e-3,
            temperature=1.0,
        ).eval()
        self.assertFalse(hasattr(mesh_read, "slot_marginal"))
        slots = torch.randn(2, 4, 16)
        context = torch.randn(2, 17, 16)
        _, diagnostics, _ = mesh_read(slots, context, collect_diagnostics=True)
        self.assertGreater(diagnostics["effective_slots"].item(), 3.5)

    def test_mesh_mix_gate_can_be_fixed_for_ablation(self):
        workspace = PersistentLatentWorkspace(
            dim=32,
            num_slots=8,
            num_heads=8,
            mlp_ratio=2.0,
            gate_min=0.05,
            gate_init=0.10,
            dropout=0.0,
            read_mode="mesh",
            mix_gate_init=0.05,
            mix_gate_trainable=False,
        )
        gate = workspace.cell.mix_gate_logit
        self.assertFalse(gate.requires_grad)
        self.assertAlmostEqual(gate.sigmoid().item(), 0.05, places=6)

        features = torch.randn(2, 256, 8, 8)
        output, latents = workspace.forward_stage("c2", features)
        (output.square().mean() + latents.square().mean()).backward()
        self.assertIsNone(gate.grad)

    def test_role_anchored_positions_and_residual_path_remain_diverse(self):
        torch.manual_seed(17)
        workspace = PersistentLatentWorkspace(
            dim=32,
            num_slots=8,
            num_heads=8,
            mlp_ratio=2.0,
            gate_min=0.05,
            gate_init=0.10,
            dropout=0.0,
            read_mode="mesh",
            role_anchor_enabled=True,
            role_anchor_scale=0.25,
            position_encoding_enabled=True,
            update_mode="residual",
            update_alphas=(0.05, 0.10, 0.15, 0.20),
            mix_enabled=False,
            ffn_layer_scale=0.10,
        ).eval()
        self.assertFalse(hasattr(workspace.cell, "mix_attention"))

        normalized = torch.nn.functional.normalize(
            _orthogonal_role_anchors(8, 32), dim=1
        )
        torch.testing.assert_close(
            normalized @ normalized.t(),
            torch.eye(8),
            rtol=1e-5,
            atol=1e-5,
        )
        position = _sincos_2d_position(
            4, 4, 32, device=torch.device("cpu"), dtype=torch.float32
        )
        self.assertEqual(tuple(position.shape), (1, 16, 32))
        self.assertGreater((position[:, 0] - position[:, -1]).abs().sum().item(), 0.0)

        stage_features = {
            "c2": torch.randn(2, 256, 8, 8),
            "c3": torch.randn(2, 512, 4, 4),
            "c4": torch.randn(2, 1024, 2, 2),
            "c5": torch.randn(2, 2048, 2, 2),
        }
        latents = None
        with torch.no_grad():
            for stage_name in workspace.stage_names:
                _, latents = workspace.forward_stage(
                    stage_name,
                    stage_features[stage_name],
                    latents,
                    collect_diagnostics=True,
                )
        diagnostics = workspace.diagnostics()
        self.assertLess(diagnostics["slot_cosine_max"].item(), 0.995)
        self.assertGreater(diagnostics["slot_effective_rank"].item(), 2.0)
        for stage_name in workspace.stage_names:
            self.assertLessEqual(
                diagnostics[f"{stage_name}_transport_entropy_delta"].item(),
                1e-3,
            )

    def test_slot_set_head_starts_with_zero_main_path_intervention(self):
        torch.manual_seed(19)
        head = _SlotSetHead(
            workspace_dim=32,
            num_slots=8,
            dim_per_slot=4,
            output_dim=64,
            num_classes=3,
            cls_type="CosSoftmax",
            scale=64.0,
            margin=0.35,
        )
        latents = torch.randn(4, 8, 32)
        targets = torch.tensor([0, 0, 1, 1])
        outputs = head(latents, targets)
        self.assertEqual(tuple(outputs["features"].shape), (4, 32))
        self.assertEqual(tuple(outputs["cls_outputs"].shape), (4, 3))
        torch.testing.assert_close(
            outputs["feature_delta"],
            torch.zeros(4, 64),
            rtol=0,
            atol=0,
        )
        head.eval()
        descriptor, feature_delta = head(latents)
        self.assertEqual(tuple(descriptor.shape), (4, 32))
        self.assertEqual(tuple(feature_delta.shape), (4, 64))

    def test_spatial_query_write_and_read_only_c5(self):
        torch.manual_seed(23)
        workspace = PersistentLatentWorkspace(
            dim=32,
            num_slots=8,
            num_heads=8,
            mlp_ratio=2.0,
            gate_min=0.0,
            gate_init=0.05,
            dropout=0.0,
            read_mode="mesh",
            role_anchor_enabled=True,
            position_encoding_enabled=True,
            update_mode="residual",
            update_alphas=(0.05, 0.10, 0.15, 0.20),
            mix_enabled=True,
            mix_gate_init=0.02,
            mix_role_conditioned=True,
            mix_centered=True,
            ffn_layer_scale=0.05,
            write_stages=("c2", "c3", "c4"),
            spatial_write_gate_enabled=True,
        ).eval()
        self.assertTrue(workspace.adapters["c2"].write_enabled)
        self.assertFalse(workspace.adapters["c5"].write_enabled)
        self.assertTrue(workspace.cell.mix_role_conditioned)
        self.assertTrue(workspace.cell.mix_centered)

        stage_features = {
            "c2": torch.randn(2, 256, 8, 8),
            "c3": torch.randn(2, 512, 4, 4),
            "c4": torch.randn(2, 1024, 2, 2),
            "c5": torch.randn(2, 2048, 2, 2),
        }
        latents = None
        outputs = {}
        with torch.no_grad():
            for stage_name in workspace.stage_names:
                outputs[stage_name], latents = workspace.forward_stage(
                    stage_name,
                    stage_features[stage_name],
                    latents,
                    collect_diagnostics=True,
                )
        self.assertFalse(torch.equal(outputs["c2"], stage_features["c2"]))
        torch.testing.assert_close(
            outputs["c5"], stage_features["c5"], rtol=0, atol=0
        )
        diagnostics = workspace.diagnostics()
        for stage_name in ("c2", "c3", "c4"):
            self.assertAlmostEqual(
                diagnostics[f"{stage_name}_gate"].item(), 0.05, places=6
            )
            self.assertIn(f"{stage_name}_gate_min", diagnostics)
            self.assertIn(f"{stage_name}_gate_max", diagnostics)
        self.assertNotIn("c5_gate", diagnostics)
        self.assertLess(diagnostics["slot_cosine_max"].item(), 0.995)
        self.assertGreater(diagnostics["slot_effective_rank"].item(), 2.0)

    def test_identity_query_head_decodes_four_views_with_gradients(self):
        torch.manual_seed(29)
        head = _IdentityQueryHead(
            workspace_dim=32,
            num_slots=8,
            num_queries=4,
            dim_per_query=8,
            num_heads=4,
            ffn_scale=0.10,
            role_anchor_scale=0.25,
            num_classes=3,
            cls_type="CosSoftmax",
            scale=64.0,
            margin=0.35,
        )
        latents = torch.randn(4, 8, 32, requires_grad=True)
        targets = torch.tensor([0, 0, 1, 1])
        outputs = head(latents, targets, collect_diagnostics=True)
        self.assertEqual(tuple(outputs["features"].shape), (4, 32))
        self.assertEqual(tuple(outputs["cls_outputs"].shape), (4, 3))
        diagnostics = head.diagnostics()
        self.assertLess(diagnostics["query_cosine_max"].item(), 0.995)
        self.assertGreater(diagnostics["query_effective_rank"].item(), 1.5)
        loss = outputs["features"].square().mean() + outputs[
            "cls_outputs"
        ].square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(latents.grad).all())
        self.assertGreater(latents.grad.abs().sum().item(), 0.0)
        for name, parameter in head.named_parameters():
            if not parameter.requires_grad:
                continue
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

        head.eval()
        with torch.no_grad():
            descriptor = head(latents.detach())
        self.assertEqual(tuple(descriptor.shape), (4, 32))

    def test_mesh_workspace_has_finite_nonzero_gradients(self):
        torch.manual_seed(13)
        workspace = self._workspace(dim=32, read_mode="mesh")
        stage_features = {
            "c2": torch.randn(2, 256, 8, 8, requires_grad=True),
            "c3": torch.randn(2, 512, 4, 4, requires_grad=True),
            "c4": torch.randn(2, 1024, 2, 2, requires_grad=True),
            "c5": torch.randn(2, 2048, 2, 2, requires_grad=True),
        }
        latents = None
        outputs = []
        for stage_name in workspace.stage_names:
            output, latents = workspace.forward_stage(
                stage_name,
                stage_features[stage_name],
                latents,
                collect_diagnostics=True,
            )
            outputs.append(output)
        (
            sum(output.square().mean() for output in outputs) + latents.square().mean()
        ).backward()

        for name, parameter in workspace.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        diagnostics = workspace.diagnostics()
        self.assertGreater(diagnostics["c5_effective_slots"].item(), 1.0)
        self.assertTrue(torch.isfinite(diagnostics["c5_assignment_entropy"]))

    def test_attention_capture_preserves_stage_shape(self):
        workspace = self._workspace(dim=32, read_mode="mesh").eval()
        workspace.set_attention_capture(True)
        features = torch.randn(1, 256, 8, 8)
        with torch.no_grad():
            workspace.forward_stage("c2", features)
        captured = workspace.attention_maps()["c2"]
        self.assertEqual(captured["spatial_size"], (4, 4))
        self.assertEqual(tuple(captured["attention"].shape), (1, 8, 16))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for AMP coverage")
    def test_mesh_read_is_finite_under_cuda_amp(self):
        mesh_read = (
            _CompetitiveMeshRead(
                dim=32,
                num_slots=8,
                mlp_ratio=2.0,
                mesh_iterations=4,
                sinkhorn_iterations=5,
                mesh_lr=1.0,
                noise_std=1e-3,
                temperature=1.0,
            )
            .cuda()
            .train()
        )
        slots = torch.randn(2, 8, 32, device="cuda", requires_grad=True)
        context = torch.randn(2, 64, 32, device="cuda", requires_grad=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output, diagnostics, _ = mesh_read(slots, context, collect_diagnostics=True)
            loss = output.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(slots.grad).all())
        self.assertTrue(torch.isfinite(context.grad).all())
        self.assertTrue(torch.isfinite(diagnostics["assignment_entropy"]))

    def test_resnest_staged_forward_is_exact(self):
        torch.manual_seed(7)
        backbone = ResNeSt(
            1,
            Bottleneck,
            [1, 1, 1, 1],
            radix=2,
            groups=1,
            bottleneck_width=64,
            deep_stem=True,
            stem_width=32,
            avg_down=True,
            avd=True,
            avd_first=False,
            norm_layer="BN",
        ).eval()
        images = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            direct = backbone(images.clone())
            staged = backbone.forward_stem(images.clone())
            for stage in range(1, 5):
                staged = backbone.forward_stage(staged, stage)
        torch.testing.assert_close(staged, direct, rtol=0, atol=0)

    def test_full_workspace_path_has_nonzero_gradients(self):
        torch.manual_seed(11)
        workspace = self._workspace()
        stage_features = {
            "c2": torch.randn(2, 256, 16, 16, requires_grad=True),
            "c3": torch.randn(2, 512, 8, 8, requires_grad=True),
            "c4": torch.randn(2, 1024, 4, 4, requires_grad=True),
            "c5": torch.randn(2, 2048, 2, 2, requires_grad=True),
        }
        latents = None
        outputs = []
        for stage_name in workspace.stage_names:
            output, latents = workspace.forward_stage(
                stage_name,
                stage_features[stage_name],
                latents,
                collect_diagnostics=True,
            )
            outputs.append(output)

        loss = sum(output.square().mean() for output in outputs)
        loss = loss + latents.square().mean()
        loss.backward()

        for name, parameter in workspace.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0, name)

        for stage_name in ("c3", "c4", "c5"):
            self.assertAlmostEqual(
                workspace.adapters[stage_name].gate().item(), 0.10, places=6
            )
        self.assertGreater(workspace.diagnostics()["slot_variance"].item(), 0.0)

    def test_workspace_checkpoint_roundtrip(self):
        workspace = self._workspace()
        original = {
            name: value.detach().clone()
            for name, value in workspace.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpointer = Checkpointer(workspace, save_dir=directory)
            checkpointer.save("workspace", epoch=3)
            with torch.no_grad():
                for parameter in workspace.parameters():
                    parameter.add_(1.0)
            metadata = checkpointer.load(str(Path(directory) / "workspace.pth"))

        self.assertEqual(metadata["epoch"], 3)
        for name, value in workspace.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)

    def test_preflight_loads_workspace_from_full_model_checkpoint(self):
        source = self._workspace(dim=32, read_mode="mesh")
        self.assertNotIn("role_anchors", source.state_dict())
        payload = {
            "model": {
                f"workspace.{name}": value.detach().clone()
                for name, value in source.state_dict().items()
            }
        }
        restored = self._workspace(dim=32, read_mode="mesh")
        with torch.no_grad():
            restored.latent_slots.add_(1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "full_model.pth"
            torch.save(payload, path)
            report = _load_workspace_weights(restored, path)
        self.assertEqual(report["loaded_tensors"], len(source.state_dict()))
        self.assertEqual(report["missing_keys"], [])
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, source.state_dict()[name])

    def test_microfit_report_accepts_learning_without_slot_collapse(self):
        records = []
        for index in range(10):
            records.append(
                {
                    "total_loss": 100.0 - 5.0 * index,
                    "latent/slot_cosine_max": 0.70 + 0.01 * index,
                    "latent/slot_effective_rank": 6.0,
                    "latent/grad_norm": 1.0,
                    "latent/grad_finite_fraction": 1.0,
                    "latent/slot_set_fusion_ratio": 0.001 * (index + 1),
                }
            )
        report = microfit_report(records, window=5, max_loss_ratio=0.85)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(all(report["checks"].values()))

    def test_health_hook_reports_live_finite_gradients(self):
        model = torch.nn.Module()
        model.workspace = self._workspace()
        model.workspace.latent_slots.square().sum().backward()
        health_hook = LatentHealthHook(model, period=1)
        health_hook.trainer = SimpleNamespace(iter=0)
        with EventStorage(0) as storage:
            health_hook.after_step()
            latest = storage.latest()

        self.assertGreater(latest["latent/grad_norm"][0], 0.0)
        self.assertGreater(latest["latent/grad_nonzero_fraction"][0], 0.0)
        self.assertEqual(latest["latent/grad_finite_fraction"][0], 1.0)
        self.assertGreater(latest["latent/parameters_with_grad_fraction"][0], 0.0)

    def test_health_hook_early_abort_requires_consecutive_bad_checks(self):
        model = torch.nn.Module()
        model.workspace = self._workspace()
        model.workspace.latent_slots.square().sum().backward()
        model.workspace._last_diagnostics["slot_cosine_max"] = torch.tensor(0.999)
        model.workspace._last_diagnostics["slot_effective_rank"] = torch.tensor(1.2)
        health_hook = LatentHealthHook(
            model,
            period=1,
            early_abort_enabled=True,
            early_abort_warmup_iters=0,
            early_abort_patience=2,
            slot_cosine_max=0.995,
            min_effective_rank=2.0,
        )
        health_hook.trainer = SimpleNamespace(iter=0)
        with EventStorage(0):
            health_hook.after_step()

        self.assertEqual(health_hook._consecutive_bad_checks, 1)
        health_hook.trainer.iter = 1
        with EventStorage(1):
            with self.assertRaisesRegex(
                RuntimeError, "Latent workspace early-abort gate"
            ):
                health_hook.after_step()
        self.assertEqual(health_hook._consecutive_bad_checks, 2)

    def test_health_hook_stops_collapsed_identity_queries(self):
        model = torch.nn.Module()
        model.workspace = self._workspace()
        model.identity_query_head = torch.nn.Linear(4, 4)
        model.fusion_weight_logit = torch.nn.Parameter(torch.tensor(0.0))
        model.workspace.latent_slots.square().sum().backward()
        model.workspace._last_diagnostics.update(
            {
                "slot_cosine_max": torch.tensor(0.8),
                "slot_effective_rank": torch.tensor(5.0),
                "identity_query_cosine_max": torch.tensor(0.999),
                "identity_query_effective_rank": torch.tensor(1.1),
            }
        )
        health_hook = LatentHealthHook(
            model,
            period=1,
            early_abort_enabled=True,
            early_abort_warmup_iters=0,
            early_abort_patience=1,
            query_cosine_max=0.995,
            min_query_rank=1.5,
        )
        health_hook.trainer = SimpleNamespace(iter=0)

        with EventStorage(0):
            with self.assertRaisesRegex(RuntimeError, "identity query"):
                health_hook.after_step()

    def test_config_build_and_freeze_boundary(self):
        cfg = get_cfg()
        add_retri_config(cfg)
        cfg.merge_from_file(str(CONFIG_ROOT / "modern_latent_workspace_smoke.yaml"))
        cfg.MODEL.BACKBONE.PRETRAIN = False
        cfg.freeze()
        model = build_model(cfg)
        self.assertEqual(model.__class__.__name__, "LatentWorkspaceBaseline")
        self.assertEqual(model.workspace.num_slots, 8)
        self.assertEqual(model.workspace.dim, 192)
        self.assertEqual(cfg.SOLVER.FREEZE_ITERS, 0)

        full_cfg = get_cfg()
        add_retri_config(full_cfg)
        full_cfg.merge_from_file(
            str(CONFIG_ROOT / "modern_latent_workspace_s101_224.yaml")
        )
        self.assertEqual(full_cfg.SOLVER.FREEZE_ITERS, 1000)

        mesh_cfg = get_cfg()
        add_retri_config(mesh_cfg)
        mesh_cfg.merge_from_file(
            str(CONFIG_ROOT / "modern_mesh_workspace_s101_224.yaml")
        )
        self.assertEqual(mesh_cfg.MODEL.LATENT_WORKSPACE.READ_MODE, "mesh")
        self.assertEqual(mesh_cfg.MODEL.LATENT_WORKSPACE.MESH_ITERS, 4)
        self.assertEqual(mesh_cfg.SEED, 20260811)

        role_cfg = get_cfg()
        add_retri_config(role_cfg)
        role_cfg.merge_from_file(str(CONFIG_ROOT / "modern_latent_role_anchored_s101_224.yaml"))
        self.assertEqual(role_cfg.MODEL.META_ARCHITECTURE, "RoleAnchoredLatentWorkspace")
        self.assertTrue(role_cfg.MODEL.LATENT_WORKSPACE.ROLE_ANCHOR_ENABLED)
        self.assertTrue(role_cfg.MODEL.LATENT_WORKSPACE.POSITION_ENCODING_ENABLED)
        self.assertEqual(role_cfg.MODEL.LATENT_WORKSPACE.UPDATE_MODE, "residual")
        self.assertFalse(role_cfg.MODEL.LATENT_WORKSPACE.MIX_ENABLED)
        self.assertTrue(role_cfg.MODEL.LATENT_WORKSPACE.SLOT_SET_HEAD_ENABLED)

        spatial_cfg = get_cfg()
        add_retri_config(spatial_cfg)
        spatial_cfg.merge_from_file(str(CONFIG_ROOT / "modern_latent_spatial_query_s101_224.yaml"))
        self.assertEqual(spatial_cfg.MODEL.META_ARCHITECTURE, "SpatialQueryLatentWorkspace")
        self.assertEqual(
            spatial_cfg.MODEL.LATENT_WORKSPACE.WRITE_STAGES, ["c2", "c3", "c4"]
        )
        self.assertTrue(
            spatial_cfg.MODEL.LATENT_WORKSPACE.SPATIAL_WRITE_GATE_ENABLED
        )
        self.assertTrue(spatial_cfg.MODEL.LATENT_WORKSPACE.MIX_ROLE_CONDITIONED)
        self.assertTrue(spatial_cfg.MODEL.LATENT_WORKSPACE.MIX_CENTERED)
        self.assertTrue(spatial_cfg.MODEL.LATENT_WORKSPACE.IDENTITY_QUERY_HEAD_ENABLED)
        self.assertFalse(spatial_cfg.MODEL.LATENT_WORKSPACE.SLOT_SET_HEAD_ENABLED)
        self.assertEqual(spatial_cfg.MODEL.LATENT_WORKSPACE.SLOT_FEATURE_FUSION_SCALE, 0.0)
        self.assertEqual(
            spatial_cfg.MODEL.LATENT_WORKSPACE.EARLY_ABORT_QUERY_COSINE_MAX, 0.995
        )
        self.assertEqual(spatial_cfg.MODEL.LATENT_WORKSPACE.EARLY_ABORT_MIN_QUERY_RANK, 1.5)

        parameter_groups = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            heads_lr_factor=cfg.SOLVER.HEADS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
            freeze_layers=cfg.MODEL.FREEZE_LAYERS,
        )
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        statuses = {"backbone": set(), "workspace": set(), "heads": set()}
        learning_rates = {"workspace": set(), "heads": set()}
        for group in parameter_groups:
            name = names[id(group["params"][0])]
            root = name.split(".")[0]
            if root in statuses:
                statuses[root].add(group["freeze_status"])
            if root in learning_rates:
                learning_rates[root].add(group["lr"])

        self.assertEqual(statuses["backbone"], {"freeze"})
        self.assertEqual(statuses["workspace"], {"normal"})
        self.assertEqual(statuses["heads"], {"normal"})
        self.assertEqual(learning_rates["workspace"], {cfg.SOLVER.BASE_LR})
        self.assertEqual(learning_rates["heads"], {cfg.SOLVER.BASE_LR})


if __name__ == "__main__":
    unittest.main()
