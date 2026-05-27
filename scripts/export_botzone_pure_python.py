#!/usr/bin/env python3
"""Export a trained MCR ensemble into a source-only Botzone Python package."""

from __future__ import annotations

import argparse
import base64
import pickle
import pprint
import shutil
import textwrap
import zipfile
from pathlib import Path


def export_hist_gradient_boosting(model) -> dict:
    """Return dependency-free data for sklearn HistGradientBoostingClassifier."""

    trees = []
    for predictor_list in model._predictors:
        tree = predictor_list[0]
        nodes = []
        for node in tree.nodes:
            nodes.append(
                [
                    int(node["feature_idx"]),
                    float(node["num_threshold"]),
                    int(node["left"]),
                    int(node["right"]),
                    float(node["value"]),
                    int(node["is_leaf"]),
                    int(node["missing_go_to_left"]),
                ]
            )
        trees.append(nodes)
    return {
        "kind": "hgb_log_loss_binary",
        "baseline": float(model._baseline_prediction[0, 0]),
        "trees": trees,
    }


def export_tfidf_sgd_pipeline(pipeline) -> dict:
    """Return dependency-free data for TfidfVectorizer + SGDClassifier."""

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    min_n, max_n = vectorizer.ngram_range
    return {
        "kind": "tfidf_sgd_log_loss_binary",
        "min_n": int(min_n),
        "max_n": int(max_n),
        "vocabulary": {str(key): int(value) for key, value in vectorizer.vocabulary_.items()},
        "idf": [float(value) for value in vectorizer.idf_],
        "coef": [float(value) for value in classifier.coef_[0]],
        "intercept": float(classifier.intercept_[0]),
    }


def load_pickle(path: Path) -> dict:
    with path.open("rb") as src:
        return pickle.load(src)


def export_payload(payload: dict) -> dict:
    if payload.get("kind") != "draw_ensemble_composite_policy":
        raise ValueError("expected draw_ensemble_composite_policy payload")
    draw_models = []
    for draw_payload in payload["draw_payloads"]:
        if draw_payload.get("kind") != "feature_action_ranker":
            raise ValueError("draw payload must be feature_action_ranker")
        draw_models.append(export_hist_gradient_boosting(draw_payload["model"]))
    reaction_payload = payload["reaction_payload"]
    if reaction_payload.get("kind") != "legal_action_ranker":
        raise ValueError("reaction payload must be legal_action_ranker")
    return {
        "kind": "pure_python_draw_ensemble",
        "prefer_hu": bool(payload.get("prefer_hu", False)),
        "draw_weights": [float(weight) for weight in payload["draw_weights"]],
        "draw_models": draw_models,
        "reaction_model": export_tfidf_sgd_pipeline(reaction_payload["pipeline"]),
    }


def write_package(model_data: dict, out_dir: Path, zip_path: Path | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_src = Path(__file__).with_name("botzone_pure_runtime.py")
    shutil.copyfile(runtime_src, out_dir / "botzone_pure_runtime.py")
    (out_dir / "botzone_model_data.py").write_text(
        "MODEL = " + pprint.pformat(model_data, width=100, sort_dicts=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "__main__.py").write_text(
        "\n".join(
            [
                "from botzone_model_data import MODEL",
                "from botzone_pure_runtime import main_with_model",
                "",
                "raise SystemExit(main_with_model(MODEL))",
                "",
            ]
        ),
        encoding="utf-8",
    )
    if zip_path is not None:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out_dir.iterdir()):
                if path.is_file():
                    archive.write(path, path.name)


def write_single_file(model_data: dict, out_path: Path) -> None:
    runtime_src = Path(__file__).with_name("botzone_pure_runtime.py")
    runtime = runtime_src.read_text(encoding="utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(
            [
                runtime.rstrip(),
                "",
                "MODEL = " + pprint.pformat(model_data, width=100, sort_dicts=True),
                "",
                "raise SystemExit(main_with_model(MODEL))",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_bootstrap_file(zip_path: Path, out_path: Path) -> None:
    encoded = base64.b64encode(zip_path.read_bytes()).decode("ascii")
    chunks = textwrap.wrap(encoded, 76)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "from __future__ import print_function",
                "",
                "import base64",
                "import io",
                "import sys",
                "import zipfile",
                "",
                "_ZIP_B64 = (",
                *['    "%s"' % chunk for chunk in chunks],
                ")",
                "",
                "def _exec_zip_module(archive, name):",
                "    filename = name + '.py'",
                "    namespace = {",
                "        '__name__': name,",
                "        '__file__': '<embedded>/' + filename,",
                "        '__package__': '',",
                "    }",
                "    source = archive.read(filename)",
                "    exec(compile(source, namespace['__file__'], 'exec'), namespace)",
                "    return namespace",
                "",
                "def _run_embedded_package():",
                "    payload = base64.b64decode(_ZIP_B64)",
                "    archive = zipfile.ZipFile(io.BytesIO(payload))",
                "    runtime_ns = _exec_zip_module(archive, 'botzone_pure_runtime')",
                "    model_ns = _exec_zip_module(archive, 'botzone_model_data')",
                "    return runtime_ns['main_with_model'](model_ns['MODEL'])",
                "",
                "raise SystemExit(_run_embedded_package())",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--zip", dest="zip_path", default=None)
    parser.add_argument("--single-file", dest="single_file", default=None)
    parser.add_argument("--bootstrap-file", dest="bootstrap_file", default=None)
    args = parser.parse_args()

    payload = load_pickle(Path(args.model))
    model_data = export_payload(payload)
    zip_path = Path(args.zip_path) if args.zip_path else None
    write_package(model_data, Path(args.out_dir), zip_path)
    print("wrote %s" % args.out_dir)
    if zip_path is not None:
        print("wrote %s" % zip_path)
    if args.single_file is not None:
        single_file = Path(args.single_file)
        write_single_file(model_data, single_file)
        print("wrote %s" % single_file)
    if args.bootstrap_file is not None:
        bootstrap_zip = zip_path
        if bootstrap_zip is None:
            bootstrap_zip = Path(args.out_dir).with_suffix(".zip")
            write_package(model_data, Path(args.out_dir), bootstrap_zip)
        bootstrap_file = Path(args.bootstrap_file)
        write_bootstrap_file(bootstrap_zip, bootstrap_file)
        print("wrote %s" % bootstrap_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
