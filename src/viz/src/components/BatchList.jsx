import { endingTurn } from "../flatten.js";

export default function BatchList({ games, startIndex, onSelect }) {
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
