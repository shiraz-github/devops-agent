import typer, json, os
from rich import print
from pathlib import Path
from .config import Settings, save_settings, load_settings, DEFAULT_CFG
from .providers import OpenAIProvider
from .index import index_repo
from .retrieve import answer

app = typer.Typer(add_completion=False)

@app.command()
def init(repo: str = typer.Argument(..., help="Path to your git repo"),
         db: str = typer.Option(None, help="Path to SQLite DB for KB")):
    db = db or str(Path(repo)/".agent/kb.sqlite")
    s = Settings(repo_path=repo, db_path=db)
    save_settings(s)
    print(f"[green]Config written to {DEFAULT_CFG}[/green]")

def _prov():
    s = load_settings()
    return s, OpenAIProvider(chat_model=s.chat_model, embedding_model=s.embedding_model)

@app.command()
def index():
    s, prov = _prov()
    sha, n = index_repo(s.repo_path, s.db_path, prov, s.max_chunk_chars, s.overlap_chars, s.exclude_globs)
    print(f"[green]Indexed {n} chunks at {sha} -> {s.db_path}[/green]")

@app.command()
def ask(q: str = typer.Argument(..., help="Your question")):
    s, prov = _prov()
    ans, hits = answer(s.db_path, prov, q)
    print("\n[bold]Answer[/bold]\n")
    print(ans)
    print("\n[bold]Citations[/bold]")
    for _, (path, st, en, _), _sc in hits[:6]:
        print(f"- {path}:{st}-{en}")

if __name__ == "__main__":
    app()

