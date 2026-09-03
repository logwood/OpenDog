"""Keep release-generation labels out of active implementation names."""

from __future__ import annotations

import ast
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
ACTIVE_SOURCE_ROOTS = (
    WORKSPACE_ROOT / "scripts",
    PROJECT_ROOT / "configs",
    PROJECT_ROOT / "fastreid",
    PROJECT_ROOT / "frontend",
    PROJECT_ROOT / "java",
    PROJECT_ROOT / "pet_id",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "tools",
    PROJECT_ROOT / "tests",
)
SOURCE_SUFFIXES = {
    ".cmd",
    ".js",
    ".jsx",
    ".ps1",
    ".py",
    ".sh",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
GENERATED_OR_DEPENDENCY_DIRECTORIES = {
    ".next",
    "dist",
    "node_modules",
    "target",
}
VERSIONED_IDENTIFIER = re.compile(
    r"(?:^|_)[vV]\d+(?:_|$)|(?<=[a-z])V\d+(?=[A-Z_]|$)"
)
VERSIONED_FILENAME = re.compile(
    r"(?i)(?:^|[_-])v\d+(?:[_-]|\.|$)|(?<=[a-z])v\d+(?=[A-Z_.-]|$)"
)
PROJECT_GENERATION_TEXT = re.compile(
    r"(?i)(?:"
    r"unified(?:[_ -]pet[_ -]reid)?|semantic|shared(?:[_ -]space)?|"
    r"latent|agent|controller|candidate|baseline|legacy"
    r")[_ -]*v\d+"
)
VERSIONED_PROJECT_ARTIFACT = re.compile(
    r"(?i)(?:dogfacenet|local_pet_gallery|pet_api_gallery|unified_pet_reid)"
    r"[a-z0-9_-]*_v\d+"
)

# These names are published third-party architecture identifiers. Renaming them
# would hide which upstream weights and tensor contracts the wrappers use.
EXTERNAL_ARCHITECTURE_IDENTIFIERS = {
    "FasterRCNN_ResNet50_FPN_V2_Weights",
    "FrozenSwinV2BodyBackbone",
    "MobileNetV2",
    "MobileNetV3",
    "ShuffleNetV2",
    "ShuffleV2Block",
    "Swin_V2_B_Weights",
    "_mobilenet_v3_conf",
    "_mobilenet_v3_model",
    "auto_augment_policy_v0",
    "fasterrcnn_resnet50_fpn_v2",
    "swin_v2_b",
}
EXTERNAL_ARCHITECTURE_FILENAMES = {"mobilenetv3.py"}


def is_active_source_path(path: Path) -> bool:
    return not any(
        part.casefold() in GENERATED_OR_DEPENDENCY_DIRECTORIES
        for part in path.parts
    )


def active_python_sources() -> list[Path]:
    return sorted(
        {
            path
            for root in ACTIVE_SOURCE_ROOTS
            if root.is_dir()
            for path in root.rglob("*.py")
            if is_active_source_path(path)
        }
    )


def identifier_candidates(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return (node.name,)
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.arg):
        return (node.arg,)
    if isinstance(node, ast.Attribute):
        return (node.attr,)
    if isinstance(node, ast.keyword) and node.arg is not None:
        return (node.arg,)
    if isinstance(node, ast.alias):
        return ((node.asname or node.name.rsplit(".", 1)[-1]),)
    return ()


def test_active_source_filenames_use_roles_or_capabilities() -> None:
    sources = {
        path
        for root in ACTIVE_SOURCE_ROOTS
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file()
        and is_active_source_path(path)
        and path.suffix.casefold() in SOURCE_SUFFIXES
    }
    sources.update(
        path
        for root in (WORKSPACE_ROOT, PROJECT_ROOT)
        for path in root.iterdir()
        if path.is_file() and path.suffix.casefold() in SOURCE_SUFFIXES
    )
    violations = [
        path.relative_to(WORKSPACE_ROOT).as_posix()
        for path in sorted(sources)
        if VERSIONED_FILENAME.search(path.name)
        and path.name not in EXTERNAL_ARCHITECTURE_FILENAMES
    ]
    assert not violations, (
        "Active source filenames must describe roles or capabilities:\n"
        + "\n".join(violations)
    )


def test_python_identifiers_do_not_encode_project_generations() -> None:
    violations: list[str] = []
    for path in active_python_sources():
        if path.name == "release_compatibility.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            for name in identifier_candidates(node):
                if (
                    VERSIONED_IDENTIFIER.search(name)
                    and name not in EXTERNAL_ARCHITECTURE_IDENTIFIERS
                ):
                    relative = path.relative_to(WORKSPACE_ROOT).as_posix()
                    violations.append(
                        f"{relative}:{getattr(node, 'lineno', 0)}: {name}"
                    )
    assert not violations, (
        "Active Python identifiers must describe roles or capabilities; "
        "put frozen release-schema aliases in release_compatibility.py:\n"
        + "\n".join(sorted(set(violations)))
    )


def test_active_source_text_does_not_name_project_generations() -> None:
    violations: list[str] = []
    sources = {
        path
        for root in ACTIVE_SOURCE_ROOTS
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file()
        and is_active_source_path(path)
        and path.suffix.casefold() in SOURCE_SUFFIXES
    }
    sources.update(
        path
        for root in (WORKSPACE_ROOT, PROJECT_ROOT)
        for path in root.iterdir()
        if path.is_file() and path.suffix.casefold() in SOURCE_SUFFIXES
    )
    for path in sorted(sources):
        if path.name in {"release_compatibility.py", Path(__file__).name}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if (
                PROJECT_GENERATION_TEXT.search(line)
                or VERSIONED_PROJECT_ARTIFACT.search(line)
            ):
                relative = path.relative_to(WORKSPACE_ROOT).as_posix()
                violations.append(f"{relative}:{line_number}: {line.strip()}")
    assert not violations, (
        "Active source text must use roles or capabilities; immutable names "
        "belong in deployment metadata or release_compatibility.py:\n"
        + "\n".join(violations)
    )
