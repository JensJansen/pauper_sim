"""Public-hostable subset of app.py: the /replay log viewer only.

app.py itself has no training-launch surface either (that used to live
here as /train + runs.py's RunManager, since removed -- training runs are
launched via run_league.py directly, CLI or the `/train` skill). What this
file additionally drops relative to app.py is the local-only server-side
log browser (/api/replay/runs, LOGS_DIR-backed) -- no filesystem access of
its own, so hosting it needs only requirements-public.txt, not the repo's
full CUDA-pinned requirements.txt. See app.py's module docstring for the
full local tool. Deploy entrypoint for Render/Fly/etc: gunicorn
src.webapp.app_public:app (see render.yaml).
"""
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling module `replay_engine`
from replay_engine import list_games, reduce_game  # noqa: E402

import json

app = Flask(__name__, static_folder="static", static_url_path="")


@app.get("/")
def replay_page():
    return send_from_directory(app.static_folder, "replay.html")


@app.post("/api/replay/games")
def replay_games():
    body = request.get_json(force=True)
    try:
        doc = json.loads(body["content"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        return jsonify({"error": f"couldn't parse log file: {exc}"}), 400
    return jsonify(list_games(doc))


@app.post("/api/replay/game")
def replay_game():
    body = request.get_json(force=True)
    try:
        doc = json.loads(body["content"])
        result = reduce_game(doc, int(body["game_index"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)
