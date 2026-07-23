import { endingTurn } from "../flatten.js";

const SEAT_LABELS = ["Agent A", "Agent B"];

function TwoPlayerBatchList({ games, startIndex, onSelect }) {
  return (
    <div className="batch-list-wrap">
      <div className="batch-header two-player">
        <div>#</div>
        <div>Game</div>
        <div>Turns</div>
        <div>Winner</div>
        <div>Agent A</div>
        <div>Agent B</div>
      </div>
      <div className="batch-list">
        {games.map((g, i) => (
          <div key={g.game_index} className="batch-row two-player" onClick={() => onSelect(g)}>
            <div className="label">#{startIndex + i + 1}</div>
            <div>Game {g.game_index}</div>
            <div className="turns">{endingTurn(g)}</div>
            <div className={`winner ${g.winner == null ? "draw" : g.winner === 0 ? "seat-a" : "seat-b"}`}>
              {g.winner == null ? "Draw" : SEAT_LABELS[g.winner]}
            </div>
            <div className="score">{Object.values(g.scores[0])[0].toFixed(3)}</div>
            <div className="score">{Object.values(g.scores[1])[0].toFixed(3)}</div>
          </div>
        ))}
        {games.length === 0 && <div className="batch-empty">No games match the current filters.</div>}
      </div>
    </div>
  );
}

export default function BatchList({ games, startIndex, onSelect, twoPlayer = false }) {
  if (twoPlayer) {
    return <TwoPlayerBatchList games={games} startIndex={startIndex} onSelect={onSelect} />;
  }

  const primaryScoreName = Object.keys(games[0]?.scores || {})[0] || "Score";
  return (
    <div className="batch-list-wrap">
      <div className="batch-header">
        <div>#</div>
        <div>Game</div>
        <div>Turns</div>
        <div>{primaryScoreName}</div>
      </div>
      <div className="batch-list">
        {games.map((g, i) => (
          <div key={g.game_index} className="batch-row" onClick={() => onSelect(g)}>
            <div className="label">#{startIndex + i + 1}</div>
            <div>Game {g.game_index}</div>
            <div className="turns">{endingTurn(g)}</div>
            <div className="score">{Object.values(g.scores)[0].toFixed(3)}</div>
          </div>
        ))}
        {games.length === 0 && <div className="batch-empty">No games match the current filters.</div>}
      </div>
    </div>
  );
}
