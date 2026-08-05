"""Local web UI for this repo's training + game-review tooling:

- Landing page (/, static/landing.html): links to the two tools below.
- Training ops (/train, static/index.html): start/stop/monitor run_league.py
  and run_pretrain.py sessions from a browser, with training_configs/*.json
  treated as optional form-prefill preconfigurations (loaded into fields
  client-side, editable after).
- Replay viewer (/replay, static/replay.html): pick a --log event-log JSON
  file from disk and step through a logged game's board state. The backend
  parses the raw log directly (replay_engine.py) -- no intermediate replay
  file format.

Serves those three static pages plus a small JSON + Server-Sent-Events API.
Local single-user tool: no auth, binds to localhost only. See README's
"Training-ops UI" section.

Run: python app.py   (from this directory, or `python src/webapp/app.py`
from anywhere -- paths are anchored to this file, not to cwd).
"""
import json
import sys
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, for `repo_paths`
from repo_paths import REPO_ROOT  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling modules `runs`, `replay_engine`
from runs import (  # noqa: E402
    LEAGUE_GLOBAL, LEAGUE_MODES, PRETRAIN_GLOBAL, PRETRAIN_SPEC,
    RunManager, argspec_from_parser, _league_parser,
)
from replay_engine import list_games, reduce_game  # noqa: E402

TRAINING_CONFIGS_DIR = REPO_ROOT / "training_configs"
LEAGUE_DECKS_PATH = REPO_ROOT / "data" / "league_decks.json"

app = Flask(__name__, static_folder="static", static_url_path="")
manager = RunManager()


@app.get("/")
def landing():
    return send_from_directory(app.static_folder, "landing.html")


@app.get("/train")
def train_page():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/argspec/<script>")
def argspec(script):
    """fields: every flag's type/help/default (script's real CLI, introspected
    for league). global: always-visible dests, regardless of mode. modes:
    (league only) collapsible groups -- see runs.py's LEAGUE_MODES docstring
    for why fields are grouped by run_league.py's real run modes."""
    if script == "pretrain":
        return jsonify({"fields": PRETRAIN_SPEC, "global": PRETRAIN_GLOBAL, "modes": []})
    if script == "league":
        return jsonify({"fields": argspec_from_parser(_league_parser()), "global": LEAGUE_GLOBAL, "modes": LEAGUE_MODES})
    return jsonify({"error": f"unknown script {script!r}"}), 404


@app.get("/api/configs")
def configs():
    """training_configs/*.json, keyed by filename -- the UI's preconfiguration
    picker. Values are only ever copied into form fields client-side, never
    referenced again once a run starts (see runs.py's module docstring)."""
    return jsonify({p.name: json.loads(p.read_text()) for p in sorted(TRAINING_CONFIGS_DIR.glob("*.json"))})


@app.get("/api/decks")
def decks():
    return jsonify(sorted(json.loads(LEAGUE_DECKS_PATH.read_text())))


@app.get("/api/runs")
def list_runs():
    return jsonify(manager.list_runs())


@app.post("/api/runs")
def start_run():
    body = request.get_json(force=True)
    run_id = manager.start(body["script"], body.get("args", {}))
    return jsonify(manager.get(run_id))


@app.get("/api/runs/<run_id>")
def run_detail(run_id):
    entry = manager.get(run_id)
    if entry is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(entry)


@app.post("/api/runs/<run_id>/stop")
def stop_run(run_id):
    return jsonify({"stopped": manager.stop(run_id)})


@app.get("/api/runs/<run_id>/log")
def run_log(run_id):
    def stream():
        for line in manager.tail_log(run_id):
            yield f"data: {json.dumps(line)}\n\n"
        yield "event: done\ndata: {}\n\n"
    return Response(stream(), mimetype="text/event-stream")


@app.get("/replay")
def replay_page():
    return send_from_directory(app.static_folder, "replay.html")


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
