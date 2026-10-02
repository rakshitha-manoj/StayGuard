"""Run the StayGuard pipeline locally, without a cluster.

    python -m pipeline.run_local --mode kfp-local [--f1-threshold 0.60]
    python -m pipeline.run_local --mode simulate  [--f1-threshold 0.60]

kfp-local : the pipeline compiled in memory (packages_to_install=[], so not byte-for-byte the
            committed YAML) and executed by KFP's own local runner
            (kfp.local.SubprocessRunner): every component runs in its own Python subprocess,
            artifacts are passed through KFP's artifact store, dsl.If is evaluated by KFP.
            It is not a cluster: no containers, no base image, no resource limits, no scheduler.
simulate  : plain Python. Calls each component's original function in order with tiny
            artifact stand-ins and evaluates the F1 condition in Python. Same function bodies,
            no KFP runtime at all.

Both build the components with packages_to_install=[] (see kfp_pipeline.build_pipeline): `src`
comes from the already-installed editable package (`pip install -e .`), so nothing is pip
installed at run time and the venv pins cannot change. Run from the repo root.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from pipeline.kfp_pipeline import COMPONENT_FUNCS, RAW_MD5, VERTEX_SKLEARN_IMAGE, build_pipeline

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "hotel_bookings.csv"
RUNS_DIR = ROOT / "pipeline" / "local_runs"
KFP_LOCAL_DIR = RUNS_DIR  # kfp.local writes <pipeline_root>/<pipeline-name>-<timestamp>/...
STEPS = ["data_collection", "data_validation", "training", "evaluation", "deployment"]


# ------------------------------------------------------------------ plain-Python simulation
class FakeArtifact:
    """Stand-in for KFP artifacts: .path, .uri, .metadata and the log_* methods components use."""

    def __init__(self, path: Path):
        self.path, self.uri, self.metadata = str(path), path.as_uri(), {}
        self.logged: dict = {}

    def log_metric(self, name, value):
        self.logged[name] = value

    def log_confusion_matrix(self, categories, matrix):
        self.logged["confusion_matrix"] = {"categories": categories, "matrix": matrix}


def simulate(args, run_dir: Path) -> dict:
    f = {name: fn for name, fn in COMPONENT_FUNCS.items()}
    status = {s: "not run" for s in STEPS}
    out: dict = {}

    def art(name):
        return FakeArtifact(run_dir / name)

    def step(name, fn, *a, **kw):
        t0 = time.perf_counter()
        try:
            res = fn(*a, **kw)
        except Exception as e:
            status[name] = f"FAILED: {type(e).__name__}: {e}"
            raise
        status[name] = f"ok ({time.perf_counter() - t0:.1f}s)"
        return res

    try:
        raw, clean, vmetrics = art("raw_data.csv"), art("clean_data.csv"), art("validation_metrics")
        model, metrics, cm, record = art("model"), art("metrics"), art("confusion_matrix"), \
            art("deployment_record.json")
        step("data_collection", f["data_collection"], args.source_uri, args.expected_md5, raw)
        step("data_validation", f["data_validation"], raw, args.adr_cap_quantile,
             args.top_n_countries, clean, vmetrics)
        step("training", f["training"], clean, args.seed, args.n_estimators,
             args.mlflow_tracking_uri, model)
        res = step("evaluation", f["evaluation"], clean, model, args.seed, metrics, cm)
        out.update(f1=res.f1, accuracy=res.accuracy, clean_md5=clean.metadata["md5"])
        if res.f1 > args.f1_threshold:  # the same condition dsl.If encodes
            step("deployment", f["deployment"], model, res.f1, args.deploy_mode,
                 VERTEX_SKLEARN_IMAGE, record)
            out["deployment_record"] = json.loads(Path(record.path).read_text())
        else:
            status["deployment"] = f"skipped (f1 {res.f1:.4f} <= threshold {args.f1_threshold})"
    except Exception:
        out["error"] = "a step failed (traceback above)"
    return {"status": status, **out}


# ------------------------------------------------------------------ KFP local runner
def _patch_windows_paths() -> None:
    """Two Windows-only workarounds for kfp 2.16's local runner, which builds an `sh -c` command
    from the executor input:
    1. `python3` is replaced with sys.executable (a backslash path on Windows); `sh` (Git Bash) eats the
       backslashes ("command not found"), so substitute the forward-slash form.
    2. Artifact paths in the executor-input JSON are built with os.path.join (backslashes), which
       breaks the JSON once `sh` unescapes it ("Invalid escape"); rewrite them with forward slashes.
       This rewrites every backslash in the executor input, parameter values included; fine here
       because no pipeline parameter legitimately contains one (Windows paths work with '/').
    Requires `sh` on PATH (Git for Windows). Nothing in KFP's files is modified; the patches live
    only in this process."""
    from kfp.local import executor_input_utils, subprocess_task_handler as h

    original_replace = h.replace_python_executable
    h.replace_python_executable = lambda cmd, exe: original_replace(cmd, Path(exe).as_posix())

    original_to_dict = executor_input_utils.executor_input_to_dict

    def to_dict(*a, **kw):
        return json.loads(json.dumps(original_to_dict(*a, **kw)).replace("\\\\", "/"))

    executor_input_utils.executor_input_to_dict = to_dict


def _patch_condition_evaluator() -> None:
    """kfp 2.16's local runner cannot evaluate the trigger condition that dsl.If compiles to:
    `inputs.parameter_values['pipelinechannel--evaluation-f1'] > inputs.parameter_values[...]`.
    Its ConditionEvaluator (kfp/local/orchestrator/enhanced_dag_orchestrator.py) has two defects,
    and any failure makes it return False, so the gated step is skipped silently:
    1. Channel names: it only recognises task outputs written `<task>--<output>` or `<task>-Output`;
       the compiler emits `<task>-<output>` (`evaluation-f1`), so it looks for a parent input,
       finds none and gives up.
    2. It substitutes values for the channel names but leaves the `inputs.parameter_values[...]`
       wrapper in the expression, so eval() raises NameError('inputs'). This hits even the most
       basic `dsl.If(op.output > 0.5)` on a single-output component (minimal repro, not
       specific to this pipeline), so it is a kfp bug, not a misuse of dsl.If here.
    This replaces evaluate_condition (this process only, kfp-local mode only) with one that
    resolves each channel as a pipeline parameter or `<task>-<output>` from the IOStore, replaces
    the whole `inputs.parameter_values['...']` lookup with the value, and evaluates the comparison.
    The compiled YAML and the component code are not affected."""
    import re

    from kfp.local.orchestrator import enhanced_dag_orchestrator as eo

    def resolve(ref: str, io_store):
        try:
            return io_store.get_parent_input(ref)
        except Exception:
            pass
        parts = ref.split("-")
        for i in range(1, len(parts)):  # try every task-name / output-name split point
            try:
                return io_store.get_task_output("-".join(parts[:i]), "-".join(parts[i:]))
            except Exception:
                continue
        raise KeyError(ref)

    def evaluate_condition(condition: str, io_store) -> bool:
        if not condition or not condition.strip():
            return True
        try:
            expr = re.sub(
                r"inputs\.parameter_values\['pipelinechannel--([^']+)'\]",
                lambda m: repr(resolve(m.group(1), io_store)), condition)
            result = bool(eval(expr, {"__builtins__": {}}, {}))
            print(f"[run_local] dsl.If condition `{expr}` -> {result}")
            return result
        except Exception as e:
            print(f"[run_local] could not evaluate condition {condition!r}: {e}")
            return False

    eo.ConditionEvaluator.evaluate_condition = staticmethod(evaluate_condition)


def run_kfp_local(args, run_dir: Path) -> dict:
    import kfp.local

    if sys.platform == "win32":
        _patch_windows_paths()
    _patch_condition_evaluator()
    pipeline, _ = build_pipeline([])
    kfp.local.init(runner=kfp.local.SubprocessRunner(use_venv=False), pipeline_root=run_dir.as_posix())
    status = {s: "not run" for s in STEPS}
    out: dict = {}
    try:
        pipeline(source_uri=args.source_uri, expected_md5=args.expected_md5,
                 f1_threshold=args.f1_threshold, seed=args.seed, n_estimators=args.n_estimators,
                 adr_cap_quantile=args.adr_cap_quantile, top_n_countries=args.top_n_countries,
                 mlflow_tracking_uri=args.mlflow_tracking_uri, deploy_mode=args.deploy_mode)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    # Read the results back from the artifact store KFP wrote under run_dir.
    def load(task_dir: str):
        for f in run_dir.rglob(f"{task_dir}/executor_output.json"):
            return json.loads(f.read_text())
        return None

    def first_artifact(doc, key):
        return doc["artifacts"][key]["artifacts"][0]

    for step, task_dir in zip(STEPS, ["data-collection", "data-validation", "training",
                                      "evaluation", "deployment"]):
        status[step] = "ok" if load(task_dir) else "not run"
    if (doc := load("data-validation")):
        out["clean_md5"] = first_artifact(doc, "clean_data")["metadata"]["md5"]
    if (doc := load("evaluation")):
        out["f1"], out["accuracy"] = doc["parameterValues"]["f1"], doc["parameterValues"]["accuracy"]
    for rec in run_dir.rglob("deployment/deployment_record"):
        out["deployment_record"] = json.loads(rec.read_text())
    if out.get("error"):  # the first step with no output is the one that failed
        failed = next((st for st in STEPS if status[st] == "not run"), None)
        if failed:
            status[failed] = "FAILED (task log above)"
    if "f1" in out and status["deployment"] == "not run":
        status["deployment"] = f"skipped (condition false: f1 {out['f1']:.4f} <= {args.f1_threshold})"
    return {"status": status, **out}


def summarize(mode: str, res: dict, args, run_dir: Path) -> None:
    print("\n" + "=" * 70)
    print(f"StayGuard pipeline summary  (mode={mode}, f1_threshold={args.f1_threshold})")
    for s in STEPS:
        print(f"  {s:<16} {res['status'][s]}")
    if "f1" in res and res["f1"] is not None:
        print(f"  f1={res['f1']:.4f}" + (f" accuracy={res['accuracy']:.4f}" if "accuracy" in res else ""))
    if res.get("clean_md5"):
        print(f"  clean_data md5 {res['clean_md5']}")
    ran = "deployment_record" in res
    print(f"  deployment ran: {ran}")
    if res.get("error"):
        print(f"  ERROR: {res['error']}")
    print(f"  run dir: {run_dir}")
    print("=" * 70)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=["kfp-local", "simulate"], required=True)
    ap.add_argument("--f1-threshold", type=float, default=0.60)
    ap.add_argument("--source-uri", default=RAW_CSV.as_posix())
    ap.add_argument("--expected-md5", default=RAW_MD5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--adr-cap-quantile", type=float, default=0.995)
    ap.add_argument("--top-n-countries", type=int, default=10)
    ap.add_argument("--mlflow-tracking-uri", default="",
                    help="e.g. the value of src.utils.MLFLOW_TRACKING_URI; empty = no MLflow")
    ap.add_argument("--deploy-mode", default="dry_run")
    args = ap.parse_args()

    run_dir = RUNS_DIR / f"{args.mode}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)
    res = simulate(args, run_dir) if args.mode == "simulate" else run_kfp_local(args, run_dir)
    summarize(args.mode, res, args, run_dir)
    return 1 if res.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
