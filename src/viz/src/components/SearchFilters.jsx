import { CARD_NAMES } from "../cardData.js";

function numOrNull(value) {
  return value === "" ? null : Number(value);
}

export default function SearchFilters({ criteria, onChange }) {
  const set = (patch) => onChange({ ...criteria, ...patch });

  return (
    <div className="search-filters">
      <div className="filter-group">
        <label>Turn won</label>
        <div className="range-inputs">
          <input
            type="number"
            min="1"
            placeholder="min"
            value={criteria.turnWonMin ?? ""}
            onChange={(e) => set({ turnWonMin: numOrNull(e.target.value) })}
          />
          <span>&ndash;</span>
          <input
            type="number"
            min="1"
            placeholder="max"
            value={criteria.turnWonMax ?? ""}
            onChange={(e) => set({ turnWonMax: numOrNull(e.target.value) })}
          />
        </div>
      </div>

      <div className="filter-group">
        <label>Card in hand</label>
        <select
          value={criteria.cardInHand ?? ""}
          onChange={(e) => set({ cardInHand: e.target.value || null })}
        >
          <option value="">Any</option>
          {CARD_NAMES.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </div>

      <div className="filter-group">
        <label>Card in play</label>
        <select
          value={criteria.cardInPlay ?? ""}
          onChange={(e) => set({ cardInPlay: e.target.value || null })}
        >
          <option value="">Any</option>
          {CARD_NAMES.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </div>

      <div className="filter-group">
        <label>Score 1</label>
        <div className="range-inputs">
          <input
            type="number"
            step="0.01"
            placeholder="min"
            value={criteria.scoreMin ?? ""}
            onChange={(e) => set({ scoreMin: numOrNull(e.target.value) })}
          />
          <span>&ndash;</span>
          <input
            type="number"
            step="0.01"
            placeholder="max"
            value={criteria.scoreMax ?? ""}
            onChange={(e) => set({ scoreMax: numOrNull(e.target.value) })}
          />
        </div>
      </div>

      <button className="clear-filters" onClick={() => onChange({})}>
        Clear filters
      </button>
    </div>
  );
}
