from pathlib import Path
from typing import Iterable
from git import Repo
from .store import connect, upsert_chunks
from .providers import OpenAIProvider

TEXT_EXT = {".py",".js",".ts",".tsx",".go",".java",".kt",".rb",".rs",".c",".h",".cpp",
            ".cs",".md",".rst",".txt",".yml",".yaml",".json",".toml",".ini",".sql",".proto",
            ".sh",".bash",".ps1",".dockerfile","Dockerfile","Makefile","Jenkinsfile"}

PIPELINE_HINTS = ("github/workflows","gitlab-ci","Jenkinsfile","circleci","azure-pipelines")

def is_text_file(p: Path) -> bool:
    if p.name in {"Jenkinsfile","Dockerfile","Makefile"}: return True
    return p.suffix.lower() in TEXT_EXT

def chunk_text(s: str, max_chars=1600, overlap=200):
    i = 0; n = len(s)
    while i < n:
        j = min(i+max_chars, n)
        # try to cut on a blank line
        k = s.rfind("\n\n", i, j)
        cut = k if k != -1 and k > i+400 else j
        yield s[i:cut]
        i = max(cut - overlap, cut)

def index_repo(repo_path: str, db_path: str, provider: OpenAIProvider,
               max_chunk_chars=1600, overlap_chars=200, exclude_globs: list[str] = None):
    exclude_globs = exclude_globs or []
    root = Path(repo_path).resolve()
    con = connect(db_path)
    repo = Repo(root)
    sha = repo.head.commit.hexsha

    files: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file(): continue
        rel = p.relative_to(root).as_posix()
        if any(Path(root/rel).match(g) for g in exclude_globs): continue
        if is_text_file(p): files.append(p)

    rows = []
    for p in files:
        rel = p.relative_to(root).as_posix()
        kind = "pipeline" if any(h in rel for h in PIPELINE_HINTS) else ("doc" if p.suffix in {".md",".rst",".txt"} else "code")
        txt = p.read_text(errors="ignore")
        line_no = 1
        chunks = list(chunk_text(txt, max_chunk_chars, overlap_chars))
        if not chunks: continue
        embs = provider.embed(chunks)
        for ch, emb in zip(chunks, embs):
            end_line = line_no + ch.count("\n")
            rows.append({"path": rel, "start_line": line_no, "end_line": end_line,
                         "sha": sha, "kind": kind, "content": ch, "embedding": emb})
            line_no = end_line + 1

    if rows:
        upsert_chunks(con, rows, embed_dim=len(rows[0]["embedding"]))
    return sha, len(rows)
