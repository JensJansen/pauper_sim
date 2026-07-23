import { useState, useEffect, useMemo, useCallback } from "react";
import Card from "./Card.jsx";
import { flattenTwoPlayerSteps } from "../flatten.js";
import {
  decisionDetailLines, formatManaPool, splitBattlefield, Zone, BattlefieldRow,
  computeBattlefieldLinks, CardPreview,
} from "./GameView.jsx";

// index-by-seat throughout, same convention harness.evaluate_two_player's
// own log (and run.py's own agent_a/agent_b model-slot naming) already
// uses -- never a dict keyed by name.
const SEAT_LABELS = ["Agent A", "Agent B"];
const SEAT_CLASS = ["seat-a", "seat-b"];

function PlayerBoard({ seat, snapshot, isTurnPlayer, isActor, onHover }) {
  const { lands, nonLands } = splitBattlefield(snapshot.battlefield);
  const links = computeBattlefieldLinks(snapshot.battlefield);
  const blocks = Object.entries(snapshot.blocked_by); // {attacker label -> blocker label}, keyed by the ATTACKER (see game/state.py's own PlayerState.blocked_by docstring)

  return (
    <div className={`player-board ${SEAT_CLASS[seat]}`}>
      <div className="player-board-header">
        <span className="player-name">{SEAT_LABELS[seat]}</span>
        {isTurnPlayer && <span className="badge badge-turn">Turn</span>}
        {isActor && <span className="badge badge-acting">Acting</span>}
        <span className="player-life">Life: {snapshot.life_total}</span>
        <span className="player-mana">Mana: {formatManaPool(snapshot.mana_pool)}</span>
      </div>

      <div className="table-upper">
        <div className="zone zone-battlefield">
          <h3>Battlefield</h3>
          <div className="battlefield-box">
            <BattlefieldRow cards={nonLands} links={links} onHover={onHover} />
            <BattlefieldRow cards={lands} links={links} onHover={onHover} />
          </div>
        </div>
        <Zone title="Graveyard" className="zone-graveyard">
          {snapshot.graveyard.map((name, i) => (
            <Card key={`${name}-${i}`} name={name} size="small" onHover={onHover} />
          ))}
        </Zone>
        <Zone title="Exile" className="zone-exile">
          {snapshot.exile.map((name, i) => (
            <Card key={`${name}-${i}`} name={name} size="small" onHover={onHover} />
          ))}
        </Zone>
      </div>

      {(snapshot.attackers.length > 0 || blocks.length > 0) && (
        <div className="combat-line">
          {snapshot.attackers.length > 0 && (
            <span>
              Attacking: <b>{snapshot.attackers.join(", ")}</b>
            </span>
          )}
          {blocks.length > 0 && (
            <span>
              Blocked: <b>{blocks.map(([attacker, blocker]) => `${attacker} by ${blocker}`).join(", ")}</b>
            </span>
          )}
        </div>
      )}

      <Zone title="Hand" className="zone-hand">
        {snapshot.hand.map((name, i) => (
          <Card key={`${name}-${i}`} name={name} size="large" onHover={onHover} />
        ))}
      </Zone>
    </div>
  );
}

export default function TwoPlayerGameView({ game, onBack }) {
  const steps = useMemo(() => flattenTwoPlayerSteps(game), [game]);
  const [stepIndex, setStepIndex] = useState(0);
  const [hoveredCard, setHoveredCard] = useState(null);

  const goto = useCallback(
    (delta) => {
      setStepIndex((i) => Math.max(0, Math.min(steps.length - 1, i + delta)));
    },
    [steps.length]
  );

  useEffect(() => {
    function onKeyDown(e) {
      if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        goto(1);
        e.preventDefault();
      } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        goto(-1);
        e.preventDefault();
      } else if (e.key === "Escape") {
        onBack();
        e.preventDefault();
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [goto, onBack]);

  const step = steps[stepIndex];
  const sa = step.state_after;
  const isLast = stepIndex === steps.length - 1;

  return (
    <div className="game-view two-player">
      {/* Left rail: nav/status chrome + a large art preview in one sticky
          column -- see GameView.jsx's own game-sidebar for the full
          rationale (shared between 1p/2p views). */}
      <div className="game-sidebar">
        <div className="back-link" onClick={onBack}>
          &larr; Back to game list (Esc)
        </div>

        <div className="step-bar">
          <div className="turn">
            Turn {step.turn_number}
            {step.turn_player_idx != null && <> &middot; {SEAT_LABELS[step.turn_player_idx]}&apos;s turn</>}
            {step.phase && <> &middot; {step.phase}</>}
          </div>
          <div className="nav">
            Step {stepIndex + 1} / {steps.length} &middot; &larr;/&rarr; to navigate
          </div>
        </div>

        <div className="action-panel">
          <div className="action-name">
            {step.actor_idx != null && (
              <span className={`actor-badge ${SEAT_CLASS[step.actor_idx]}`}>{SEAT_LABELS[step.actor_idx]}</span>
            )}
            {step.action}
            {step.fallback && (
              <span
                className="fallback-badge"
                title="Model chose an illegal action here; the first legal action was substituted."
              >
                fallback
              </span>
            )}
          </div>
          {step.fetched?.length > 0 && (
            <div className="detail-line">
              Fetched: <b>{step.fetched.join(", ")}</b>
            </div>
          )}
          {step.left_battlefield?.length > 0 && (
            <div className="detail-line">
              Left battlefield: <b>{step.left_battlefield.join(", ")}</b>
            </div>
          )}
          {step.tapped_for_cost?.length > 0 && (
            <div className="detail-line">
              Tapped to pay for it: <b>{step.tapped_for_cost.join(", ")}</b>
            </div>
          )}
          {decisionDetailLines(step.decision).map(([label, value]) => (
            <div className="detail-line" key={label}>
              {label}: <b>{value}</b>
            </div>
          ))}
        </div>

        <div className="card-preview-panel">
          <CardPreview hovered={hoveredCard} />
        </div>
      </div>

      <div className="game-view-main">
        {isLast && (
          <div className={`outcome ${game.winner == null ? "draw" : "success"}`}>
            <div className="headline">{game.winner == null ? "Draw" : `${SEAT_LABELS[game.winner]} wins`}</div>
            <div className="sub">
              {game.winner == null
                ? "Safety-cap horizon reached, no winner"
                : `Won turn ${game.turn_won ?? game.final_turn_number}`}
            </div>
            <div className="outcome-scores">
              {SEAT_LABELS.map((label, seat) => (
                <span key={seat}>
                  <b>{label}</b>:{" "}
                  {Object.entries(game.scores[seat]).map(([name, s]) => `${name} ${s.toFixed(2)}`).join(", ")}
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="two-player-boards">
          {sa.players.map((snapshot, seat) => (
            <PlayerBoard
              key={seat}
              seat={seat}
              snapshot={snapshot}
              isTurnPlayer={step.turn_player_idx === seat}
              isActor={step.actor_idx === seat}
              onHover={setHoveredCard}
            />
          ))}
        </div>

        {sa.stack.length > 0 && (
          <div className="detail-line">
            Stack (top resolves next, listed last): <b>{sa.stack.join(", ")}</b>
          </div>
        )}
      </div>
    </div>
  );
}
