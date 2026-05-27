import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_transformer_chaga_review import select_reviewed_examples  # noqa: E402


def test_select_reviewed_examples_keeps_only_examples_with_teacher_distribution():
    examples = [
        SimpleNamespace(teacher_action_distribution=None),
        SimpleNamespace(teacher_action_distribution=[1.0, 0.0]),
        SimpleNamespace(teacher_action_distribution=None),
    ]

    assert select_reviewed_examples(examples) == [examples[1]]
