from pathlib import Path
import yaml

def summarize_github_actions(flow_path: Path) -> dict:
    data = yaml.safe_load(flow_path.read_text())
    jobs = data.get("jobs", {})
    out = {"name": data.get("name", flow_path.name), "on": data.get("on"), "jobs": []}
    for jname, j in jobs.items():
        steps = [s.get("name") or next(iter(s.keys())) for s in j.get("steps", [])]
        out["jobs"].append({"name": jname, "runs-on": j.get("runs-on"), "steps": steps})
    return out
