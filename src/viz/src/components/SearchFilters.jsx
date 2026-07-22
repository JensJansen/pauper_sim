import { CARD_NAMES } from "../cardData.js";

function numOrNull(value) {
  return value === "" ? null : Number(value);
}

function RangeFilter({ label, min, step, minValue, maxValue, onMinChange, onMaxChange }) {
  return (
    <div className="filter-group">
      <label>{label}</label>
      <div className="range-inputs">
        <input
          type="number"
          min={min}
          step={step}
          placeholder="min"
          value={minValue ?? ""}
          onChange={(e) => onMinChange(numOrNull(e.target.value))}
        />
        <span>&ndash;</span>
        <input
          type="number"
          min={min}
          step={step}
          placeholder="max"
          value={maxValue ?? ""}
          onChange={(e) => onMaxChange(numOrNull(e.target.value))}
        />
      </div>
    </div>
  );
}

function CardSelect({ label, value, onChange }) {
  return (
    <div className="filter-group">
      <label>{label}</label>
      <select value={value ?? ""} onChange={(e) => onChange(e.target.value || null)}>
        <option value="">Any</option>
        {CARD_NAMES.map((name) => (
          <option key={name} value={name}>
            {name}
          </option>
        ))}
      </select>
    </div>
  );
}

export default function SearchFilters({ criteria, onChange }) {
  const set = (patch) => onChange({ ...criteria, ...patch });

  return (
    <div className="search-filters">
      <RangeFilter
        label="Turn won"
        min="1"
        minValue={criteria.turnWonMin}
        maxValue={criteria.turnWonMax}
        onMinChange={(v) => set({ turnWonMin: v })}
        onMaxChange={(v) => set({ turnWonMax: v })}
      />

      <CardSelect label="Card in hand" value={criteria.cardInHand} onChange={(v) => set({ cardInHand: v })} />

      <CardSelect label="Card in play" value={criteria.cardInPlay} onChange={(v) => set({ cardInPlay: v })} />

      <RangeFilter
        label="Score 1"
        step="0.01"
        minValue={criteria.scoreMin}
        maxValue={criteria.scoreMax}
        onMinChange={(v) => set({ scoreMin: v })}
        onMaxChange={(v) => set({ scoreMax: v })}
      />

      <button className="clear-filters" onClick={() => onChange({})}>
        Clear filters
      </button>
    </div>
  );
}
