"""Resolve the released TokenMatch core and shared evaluation helpers."""
import hashlib
import importlib.util
from pathlib import Path
import re
import sys
from types import SimpleNamespace

DEFAULT_PROJECT = str(Path(__file__).resolve().parent.parent)
EXPERIMENT = "20240815tokenmatch_up_new_stran_msg0.6-0.9-softmax51w"
EXPECTED_CORE_SHA256 = "54467c755baa0584bea33dbe449fd102a6d3c4ba293c01d6afbb32b2bae27a1b"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluator(project):
    """Load the independent relative-pose evaluator, without matcher-specific code."""
    filename = Path(project).resolve() / "TokenMatch_visual_integration" / "pose_evaluation.py"
    if not filename.is_file():
        raise FileNotFoundError("Missing shared pose evaluator: " + str(filename))
    spec = importlib.util.spec_from_file_location("tokenmatch_pose_evaluation", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_core(project):
    core = Path(project).resolve() / "TokenMatch" / "core"
    filename = core / "tokenmatch.py"
    if not filename.is_file():
        raise FileNotFoundError("Missing TokenMatch backbone: " + str(filename))
    if sha256(filename) != EXPECTED_CORE_SHA256:
        raise RuntimeError("Backbone differs from the verified release version: " + str(filename))
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))
    import tokenmatch as module
    # Preserve the CPU eigensolve and restore the eigenvectors to the input device.
    import torch
    module.batch_symeig = lambda x: torch.linalg.eigh(x.cpu(), UPLO="U").eigenvectors.to(x.device)
    return module


def configuration():
    # Configuration used by the paired training/evaluation integration.
    return SimpleNamespace(net_channels=128, cluster=64, iter_num=1,
                           use_ratio=0, use_mutual=0, use_fundamental=False,
                           loss_essential=0.5, loss_classif=1.0, obj_geod_th=1e-4,
                           geo_loss_margin=0.1, loss_essential_init_iter=20000)


def fresh_model(project, variant, config=None):
    core = setup_core(project)
    config = config or configuration()
    if variant == "visual":
        from tokenmatch_visual_additive_updated import Tokenmatch
        return Tokenmatch(config)
    if variant == "coordinate":
        return core.Tokenmatch(config)
    raise ValueError("Unknown variant " + str(variant))


def load_checkpoint(project, path, variant, device, original=False):
    import torch
    # Only load checkpoint files from trusted sources.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    state = {k.removeprefix("module."): v for k, v in state.items()
             if not k.endswith(("total_ops", "total_params"))}
    if variant == "visual" and not any(k.startswith("visual_encoder.") for k in state):
        raise RuntimeError("Coordinate-only checkpoint cannot serve as a trained visual model.")
    config = configuration()
    indices = [int(m.group(1)) for key in state
               if (m := re.match(r'weights_iter\.(\d+)\.', key))]
    config.iter_num = max(indices, default=-1) + 1
    model = fresh_model(project, variant, config)
    model.load_state_dict(state, strict=True)
    return model.to(device), checkpoint


def provenance(project):
    project = Path(project).resolve()
    paths = {
        "core": project / "TokenMatch/core/tokenmatch.py",
        "loss": project / "TokenMatch/core/loss.py",
        "train_reference": project / "TokenMatch/core/train.py",
        "evaluation": project / "TokenMatch_visual_integration/pose_evaluation.py",
    }
    return {key: {"path": str(path.relative_to(project)), "sha256": sha256(path)} for key, path in paths.items()}


def integration_hashes():
    folder = Path(__file__).resolve().parent
    return {p.name: sha256(p) for p in sorted(folder.glob('*.py'))}
