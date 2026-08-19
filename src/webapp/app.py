"""Local web UI for this repo's game-review tooling: the replay viewer
(/, static/replay.html) -- pick a --log event-log JSON file from disk, or
browse logs/<run>/event_log.json files already on disk (/api/replay/runs,
local-only), and step through a logged game's board state. The backend
parses the raw log directly (replay_engine.py) -- no intermediate replay
file format.

Local single-user tool: no auth, binds to localhost only. See README's
"Game replay viewer" section. Training runs are launched via run_league.py
directly (CLI, or the `/train` skill) -- this app has no training-launch
surface; see app_public.py for the publicly-hostable subset of this same
viewer.

Run: python app.py   (from this directory, or `python src/webapp/app.py`
from anywhere -- paths are anchored to this file, not to cwd).
"""
import json
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, for `repo_paths`
from repo_paths import REPO_ROOT  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling module `replay_engine`
from replay_engine import list_games, reduce_game  # noqa: E402

LOGS_DIR = REPO_ROOT / "logs"

app = Flask(__name__, static_folder="static", static_url_path="")


@app.get("/")
def replay_page():
    return send_from_directory(app.static_folder, "replay.html")


@app.get("/api/replay/runs")
def replay_runs():
    """Server-side log browser -- local-only (app_public.py has no equivalent,
    see its own docstring). Every logs/<run>/event_log.json on disk, newest
    first -- however it got there (a --log PATH pointed at that folder, by
    hand or from a script/skill following the same convention)."""
    runs = []
    for event_path in LOGS_DIR.glob("*/event_log.json"):
        stat = event_path.stat()
        runs.append({"name": event_path.parent.name, "mtime": stat.st_mtime, "size_kb": stat.st_size / 1024})
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return jsonify(runs)


@app.get("/api/replay/runs/<name>/raw")
def replay_run_raw(name):
    if not (LOGS_DIR / name / "event_log.json").is_file():
        return jsonify({"error": "not found"}), 404
    return send_from_directory(LOGS_DIR / name, "event_log.json")


@app.post("/api/replay/games")
def replay_games():
    """Body: {"content": <raw log JSON text, read client-side from a user-picked
    file>}. Returns the game index for that file (label + event count per
    game) without reducing any board state -- cheap even for a multi-
    thousand-game round-robin --eval log."""
    body = request.get_json(force=True)
    try:
        doc = json.loads(body["content"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        return jsonify({"error": f"couldn't parse log file: {exc}"}), 400
    return jsonify(list_games(doc))


@app.post("/api/replay/game")
def replay_game():
    """Body: {"content": <same raw log JSON text>, "game_index": N}. Returns
    one board-state snapshot per event in that game."""
    body = request.get_json(force=True)
    try:
        doc = json.loads(body["content"])
        result = reduce_game(doc, int(body["game_index"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)
