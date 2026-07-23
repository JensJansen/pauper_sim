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
// choose_opponent_permanent/declare_blockers are 2-player/combat-only
// (blocking's own nested "which attacker" consult); the rest can appear in
// either a 1- or 2-player log.
const SIMPLE_DECISION_FIELDS = {
  search_fetch: ["Library cards matching the search", "library_matches"],
  choose_permanent: ["Eligible permanents", "battlefield_matches"],
  discard: ["Hand options", "hand_options"],
  sacrifice: ["Eligible to sacrifice", "sacrifice_options"],
  mulligan_decision: ["Mulligan options", "options"],
  mulligan_bottom: ["Cards eligible to bottom", "bottom_options"],
  order_triggers: ["Triggers to place", "trigger_options"],
  choose_opponent_permanent: ["Opponent's eligible permanents", "opponent_battlefield_matches"],
  declare_blockers: ["Attackers still needing a blocker", "unblocked_attackers"],
};

export function decisionDetailLines(decision) {
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

export function formatManaPool(manaPool) {
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

export function splitBattlefield(battlefield) {
  // An enchanting Aura (p.enchanting set) is excluded here -- it doesn't
  // get its own row slot at all, it renders stacked on its target instead
  // (see BattlefieldRow/computeBattlefieldLinks' own enchantingAuras), even
  // when the Aura's own type would otherwise sort it into a different row
  // than its target (e.g. Utopia Sprawl, an ENCHANTMENT, enchanting a
  // Forest that lands in the lands row).
  const sorted = sortedBattlefield(battlefield.filter((p) => !p.enchanting));
  return {
    nonLands: sorted.filter((p) => cardInfo(p.name).type !== "LAND"),
    lands: sorted.filter((p) => cardInfo(p.name).type === "LAND"),
  };
}

function permanentLabel(p) {
  // Matches harness._snapshot_player_state's own "Name (slot k)" format
  // for an Aura's "enchanting" target exactly, so a straight string lookup
  // (not name+slot field comparison) finds the right permanent -- p.slot
  // is undefined for a 1-player log's battlefield entries (_snapshot_state
  // never included it), which degrades this to a bare name there, the
  // same as every entry already reads without slot disambiguation.
  return p.slot != null ? `${p.name} (slot ${p.slot})` : p.name;
}

// One seat's WHOLE battlefield (not yet split into lands/non-lands) in --
// both the "is this name ambiguous" count and the "who enchants me" lookup
// need every permanent regardless of which land/non-land row it ends up
// in (an enchanted Forest's own Aura, e.g. Utopia Sprawl, is a non-land
// sharing a row with creatures, while its target sits in the lands row).
// Called once per battlefield per render (GameView/TwoPlayerGameView),
// shared across both of that battlefield's own BattlefieldRow calls.
export function computeBattlefieldLinks(battlefield) {
  const nameCounts = {};
  for (const p of battlefield) {
    nameCounts[p.name] = (nameCounts[p.name] || 0) + 1;
  }
  const enchantedBy = {};
  const enchantingAuras = {};
  for (const p of battlefield) {
    if (p.enchanting) {
      if (!enchantedBy[p.enchanting]) enchantedBy[p.enchanting] = [];
      enchantedBy[p.enchanting].push(p.name);
      // Full permanent objects (not just names), keyed by the TARGET's own
      // label -- BattlefieldRow renders these stacked on the target
      // directly, regardless of which row the Aura's own type would
      // otherwise have sorted it into (see splitBattlefield above).
      if (!enchantingAuras[p.enchanting]) enchantingAuras[p.enchanting] = [];
      enchantingAuras[p.enchanting].push(p);
    }
  }
  return { nameCounts, enchantedBy, enchantingAuras };
}

export function BattlefieldRow({ cards, links, onHover }) {
  return (
    <div className="battlefield-row">
      {cards.map((p, i) => {
        // Any Aura enchanting THIS permanent, wherever it landed in the
        // (name, slot) label space -- stacked behind it below, peeking out
        // top-right, rather than occupying its own row slot (reference:
        // the user's example app stacks an equip/enchant relationship this
        // exact way, offset-behind rather than side-by-side).
        const auras = links?.enchantingAuras[permanentLabel(p)] || [];
        return (
          <div className="permanent-stack" key={`${p.name}-${i}`}>
            {auras.map((aura, j) => (
              <div className="stacked-aura" style={{ "--stack-i": j }} key={`${aura.name}-${j}`}>
                <Card
                  name={aura.name}
                  tapped={aura.tapped}
                  slot={links.nameCounts[aura.name] > 1 ? aura.slot : null}
                  enchanting={aura.enchanting}
                  onHover={onHover}
                />
              </div>
            ))}
            <Card
              name={p.name}
              tapped={p.tapped}
              // Slot badge only when this name is actually ambiguous (more
              // than one copy on this same battlefield) -- avoids clutter on
              // the common singleton case while still disambiguating exactly
              // when it matters (docs: two same-named permanents, one tapped
              // from an earlier attack and unrelated to the other dying as a
              // blocker, is exactly the case that reads wrong without this).
              slot={links && links.nameCounts[p.name] > 1 ? p.slot : null}
              power={p.power}
              toughness={p.toughness}
              basePower={p.base_power}
              baseToughness={p.base_toughness}
              keywords={p.keywords}
              enchantedBy={links?.enchantedBy[permanentLabel(p)]}
              onHover={onHover}
            />
          </div>
        );
      })}
    </div>
  );
}

// Sidebar detail for whatever card is currently hovered -- `hovered` is
// the {name, enchanting, enchantedBy} shape Card.jsx's own onHover always
// reports (null for every non-battlefield card, since Card only ever
// receives those two props from a battlefield permanent -- see
// BattlefieldRow above). Shared by GameView/TwoPlayerGameView so the two
// don't duplicate this rendering.
export function CardPreview({ hovered }) {
  if (!hovered) return null;
  // Only creature battlefield entries carry power/toughness at all (see
  // harness._snapshot_player_state) -- power != null is the presence
  // check. Only call out "(base X/Y)" when an Aura bonus actually changed
  // something; otherwise the base numbers ARE the current numbers and
  // repeating them is just noise.
  const isBuffed = hovered.power != null && (hovered.power !== hovered.basePower || hovered.toughness !== hovered.baseToughness);
  return (
    <>
      <Card name={hovered.name} width={260} />
      {hovered.power != null && (
        <div className="preview-stat-line">
          {hovered.power}/{hovered.toughness}
          {isBuffed && (
            <span className="preview-stat-base"> (base {hovered.basePower}/{hovered.baseToughness})</span>
          )}
        </div>
      )}
      {hovered.keywords?.length > 0 && <div className="preview-stat-line">{hovered.keywords.join(", ")}</div>}
      {hovered.enchanting && (
        <div className="preview-enchant-line">
          Enchanting: <b>{hovered.enchanting}</b>
        </div>
      )}
      {hovered.enchantedBy?.length > 0 && (
        <div className="preview-enchant-line">
          Enchanted by: <b>{hovered.enchantedBy.join(", ")}</b>
        </div>
      )}
    </>
  );
}

export function Zone({ title, children, className = "" }) {
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
  const links = computeBattlefieldLinks(sa.battlefield);
  // Only meaningful for a damage-race deck (meta.win_threshold set --
  // see terminated.damage_threshold_terminated); null for e.g. Tron,
  // which has no such notion, and for older logs predating this field.
  const opponentLife =
    meta?.win_threshold != null && sa.damage_dealt != null
      ? Math.max(0, meta.win_threshold - sa.damage_dealt)
      : null;

  return (
    <div className="game-view">
      {/* Left rail: navigation + status chrome (back link, turn/step
          counter, current action) plus a large art preview, all stacked in
          one sticky column -- kept together so scrolling the board below
          never carries them out of view (report: the turn/step counter
          used to scroll off screen entirely), and so the board itself gets
          the freed-up width instead of two skinny side margins. */}
      <div className="game-sidebar">
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

        {/* Always present (empty when nothing's hovered) so hovering never
            shifts the rest of the sidebar. */}
        <div className="card-preview-panel">
          <CardPreview hovered={hoveredCard} />
        </div>
      </div>

      <div className="game-view-main">
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
                <BattlefieldRow cards={nonLands} links={links} onHover={setHoveredCard} />
                <BattlefieldRow cards={lands} links={links} onHover={setHoveredCard} />
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
    </div>
  );
}
