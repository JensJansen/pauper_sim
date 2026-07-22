import { useState, useEffect, useMemo, useCallback } from "react";
import Card from "./Card.jsx";
import { flattenSteps } from "../flatten.js";
import { cardInfo } from "../cardData.js";

// One line per piece of hidden information a resolution step's decision
// was actually based on -- generic over every game.pending_resolution
// kind harness.py's _snapshot_pending can produce, rather than one
// component per kind, since they're all just "label: list of names".
// kind -> [label, field]: every resolution kind whose detail is just "the
// list of names the model was choosing among," rendered identically.
const SIMPLE_DECISION_FIELDS = {
  search_fetch: ["Library cards matching the search", "library_matches"],
  choose_permanent: ["Eligible permanents", "battlefield_matches"],
  discard: ["Hand options", "hand_options"],
  sacrifice: ["Eligible to sacrifice", "sacrifice_options"],
};

function decisionDetailLines(decision) {
  if (!decision) return [];
  if (decision.kind === "scry" || decision.kind === "surveil") {
    return [
      decision.current_card && ["Deciding", decision.current_card],
      decision.remaining.length > 0 && ["Still to reveal", decision.remaining.join(", ")],
      decision.kept.length > 0 && ["Kept on top", decision.kept.join(", ")],
      decision.disposed.length > 0 && [decision.kind === "scry" ? "Bottomed" : "Binned", decision.disposed.join(", ")],
    ].filter(Boolean);
  }
  if (decision.kind === "madness_decision") {
    return [["Cast for madness cost or let go to graveyard", decision.card]];
  }
  const simple = SIMPLE_DECISION_FIELDS[decision.kind];
  return simple ? [[simple[0], decision[simple[1]].join(", ") || "none"]] : [];
}

function formatManaPool(manaPool) {
  const entries = Object.entries(manaPool);
  if (entries.length === 0) return "none";
  return entries.map(([color, count]) => `${color}:${count}`).join(", ");
}

export function sortedBattlefield(battlefield) {
  // Stable sort (spec-guaranteed since ES2019): ties keep their original
  // order. Lands first, then artifacts, then creatures -- real MTG table
  // convention, closest-to-viewer to furthest.
  return [...battlefield].sort(
    (a, b) => cardInfo(a.name).sortPriority - cardInfo(b.name).sortPriority
  );
}

function splitBattlefield(battlefield) {
  const sorted = sortedBattlefield(battlefield);
  return {
    nonLands: sorted.filter((p) => cardInfo(p.name).type !== "LAND"),
    lands: sorted.filter((p) => cardInfo(p.name).type === "LAND"),
  };
}

function BattlefieldRow({ cards, onHover }) {
  return (
    <div className="battlefield-row">
      {cards.map((p, i) => (
        <Card key={`${p.name}-${i}`} name={p.name} tapped={p.tapped} onHover={onHover} />
      ))}
    </div>
  );
}

function Zone({ title, children, className = "" }) {
  const empty = Array.isArray(children) ? children.length === 0 : !children;
  return (
    <div className={`zone ${className}`}>
      <h3>{title}</h3>
      <div className="zone-cards">{empty ? <span className="empty">empty</span> : children}</div>
    </div>
  );
}

export default function GameView({ game, meta, onBack }) {
  const steps = useMemo(() => flattenSteps(game), [game]);
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
  const { lands, nonLands } = splitBattlefield(sa.battlefield);
  // Only meaningful for a damage-race deck (meta.win_threshold set --
  // see terminated.damage_threshold_terminated); null for e.g. Tron,
  // which has no such notion, and for older logs predating this field.
  const opponentLife =
    meta?.win_threshold != null && sa.damage_dealt != null
      ? Math.max(0, meta.win_threshold - sa.damage_dealt)
      : null;

  return (
    <div className="game-view">
      <div className="game-view-main">
        <div className="back-link" onClick={onBack}>
          &larr; Back to game list (Esc)
        </div>

        <div className="step-bar">
          <div className="turn">Turn {sa.turn_number}</div>
          {opponentLife != null && <div className="opponent-life">Opponent life: {opponentLife}</div>}
          <div className="nav">
            Step {stepIndex + 1} / {steps.length} &middot; &larr;/&rarr; to navigate
          </div>
        </div>

        {isLast && (
          <div className={`outcome ${game.turn_won !== null ? "success" : "fail"}`}>
            <div className="headline">
              {game.turn_won !== null ? "Success" : "Failure"} &middot;{" "}
              {Object.entries(game.scores).map(([name, s]) => `${name}: ${s.toFixed(2)}`).join(", ")}
            </div>
            <div className="sub">
              {game.turn_won !== null ? `Won turn ${game.turn_won}` : "Never won"}
            </div>
          </div>
        )}

        <div className="action-panel">
          <div className="action-name">
            {step.action}
            {step.fallback && (
              <span className="fallback-badge" title="Model chose an illegal action here; the first legal action was substituted.">
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

        {/* Table layout, top to bottom: battlefield (the play area) is the
            main surface; graveyard is a smaller, secondary pile off to the
            side of it; hand sits at the bottom, closest to the viewer, in
            the largest cards on the page -- matching how an actual game of
            Magic is laid out on screen (battlefield above, hand below). */}
        <div className="table">
          <div className="table-upper">
            <div className="zone zone-battlefield">
              <h3>Battlefield</h3>
              <div className="battlefield-box">
                <BattlefieldRow cards={nonLands} onHover={setHoveredCard} />
                <BattlefieldRow cards={lands} onHover={setHoveredCard} />
              </div>
            </div>
            <Zone title="Discard pile" className="zone-graveyard">
              {sa.graveyard.map((name, i) => (
                <Card key={`${name}-${i}`} name={name} size="small" onHover={setHoveredCard} />
              ))}
            </Zone>
            <Zone title="Exile" className="zone-exile">
              {sa.exile?.map((name, i) => (
                <Card key={`${name}-${i}`} name={name} size="small" onHover={setHoveredCard} />
              ))}
            </Zone>
          </div>

          <div className="resource-quality">
            <span>
              Non-land permanents: <b>{sa.resource_quality.non_land_permanents}</b>
            </span>
            <span>
              Available mana: <b>{sa.resource_quality.available_mana}</b>
            </span>
            <span>
              Hand size: <b>{sa.resource_quality.hand_size}</b>
            </span>
            <span>
              Floating mana: <b>{formatManaPool(sa.mana_pool)}</b>
            </span>
          </div>

          <Zone title="Hand" className="zone-hand">
            {sa.hand.map((name, i) => (
              <Card key={`${name}-${i}`} name={name} size="large" onHover={setHoveredCard} />
            ))}
          </Zone>
        </div>
      </div>

      {/* Fixed-width sidebar, always present so hovering never shifts
          layout -- shows a larger, easier-to-read version of whatever
          card is currently under the mouse. */}
      <div className="card-preview-panel">
        {hoveredCard && <Card name={hoveredCard} width={260} />}
      </div>
    </div>
  );
}
