import { useState, useEffect, useMemo, useCallback } from "react";
import Card from "./Card.jsx";
import { flattenSteps } from "../flatten.js";
import { cardInfo } from "../cardData.js";

export function sortedBattlefield(battlefield) {
  // Stable sort (spec-guaranteed since ES2019): ties keep their original
  // order. Lands first, then artifacts, then creatures -- real MTG table
  // convention, closest-to-viewer to furthest.
  return [...battlefield].sort(
    (a, b) => cardInfo(a.name).sortPriority - cardInfo(b.name).sortPriority
  );
}

// Battlefield box is a fixed 5-wide x 2-row grid (lands bottom row, every
// other permanent type above) that never resizes. A row that would exceed
// 5 cards shrinks its card width to still fit, rather than wrapping.
const BF_COLS = 5;
const BF_CARD_WIDTH = 90;
const BF_GAP_X = 10;
const BF_ROW_WIDTH = BF_COLS * BF_CARD_WIDTH + (BF_COLS - 1) * BF_GAP_X;

function battlefieldRowCardWidth(count) {
  if (count <= BF_COLS) return BF_CARD_WIDTH;
  const shrunk = (BF_ROW_WIDTH - (count - 1) * BF_GAP_X) / count;
  return Math.max(shrunk, 20); // ponytail: floor so cards never vanish at extreme counts
}

function splitBattlefield(battlefield) {
  const sorted = sortedBattlefield(battlefield);
  return {
    nonLands: sorted.filter((p) => cardInfo(p.name).type !== "LAND"),
    lands: sorted.filter((p) => cardInfo(p.name).type === "LAND"),
  };
}

function BattlefieldRow({ cards, onHover }) {
  const width = battlefieldRowCardWidth(cards.length);
  return (
    <div className="battlefield-row">
      {cards.map((p, i) => (
        <Card key={`${p.name}-${i}`} name={p.name} tapped={p.tapped} width={width} onHover={onHover} />
      ))}
    </div>
  );
}

function Zone({ title, children, empty, className = "" }) {
  return (
    <div className={`zone ${className}`}>
      <h3>{title}</h3>
      <div className="zone-cards">{empty ? <span className="empty">empty</span> : children}</div>
    </div>
  );
}

export default function GameView({ game, onBack }) {
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

  return (
    <div className="game-view">
      <div className="game-view-main">
        <div className="back-link" onClick={onBack}>
          &larr; Back to game list (Esc)
        </div>

        <div className="step-bar">
          <div className="turn">Turn {sa.turn_number}</div>
          <div className="nav">
            Step {stepIndex + 1} / {steps.length} &middot; &larr;/&rarr; to navigate
          </div>
        </div>

        {isLast && (
          <div className={`outcome ${game.turn_won !== null ? "success" : "fail"}`}>
            <div className="headline">
              {game.turn_won !== null ? "Success" : "Failure"} &middot;{" "}
              {game.scores.map((s, i) => `score ${i + 1}: ${s.toFixed(2)}`).join(", ")}
            </div>
            <div className="sub">
              {game.turn_won !== null ? `Won turn ${game.turn_won}` : "Never won"}
            </div>
          </div>
        )}

        <div className="action-panel">
          <div className="action-name">{step.action}</div>
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
          {step.scry && (
            <div className="detail-line">
              Looked at: <b>{step.scry.seen.join(", ")}</b>
              {step.scry.kept_on_top.length > 0 && (
                <>
                  {" "}
                  &middot; kept on top: <b>{step.scry.kept_on_top.join(", ")}</b>
                </>
              )}
              {step.scry.bottomed.length > 0 && (
                <>
                  {" "}
                  &middot; bottomed: <b>{step.scry.bottomed.join(", ")}</b>
                </>
              )}
              {step.scry.binned.length > 0 && (
                <>
                  {" "}
                  &middot; discarded: <b>{step.scry.binned.join(", ")}</b>
                </>
              )}
            </div>
          )}
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
            <Zone title="Discard pile" className="zone-graveyard" empty={sa.graveyard.length === 0}>
              {sa.graveyard.map((name, i) => (
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
          </div>

          <Zone title="Hand" className="zone-hand" empty={sa.hand.length === 0}>
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
