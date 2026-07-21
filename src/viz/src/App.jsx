import { useState, useMemo, useCallback } from "react";
import DropZone from "./components/DropZone.jsx";
import SearchFilters from "./components/SearchFilters.jsx";
import Pagination from "./components/Pagination.jsx";
import BatchList from "./components/BatchList.jsx";
import GameView from "./components/GameView.jsx";
import { filterGames } from "./filterGames.js";
import "./App.css";

const PAGE_SIZE = 25;

export default function App() {
  const [games, setGames] = useState(null);
  const [meta, setMeta] = useState(null);
  const [fileName, setFileName] = useState(null);
  const [criteria, setCriteria] = useState({});
  const [page, setPage] = useState(1);
  const [selectedGame, setSelectedGame] = useState(null);

  const handleLoad = useCallback(({ meta: loadedMeta, games: loadedGames }, name) => {
    setGames(loadedGames);
    setMeta(loadedMeta);
    setFileName(name);
    setCriteria({});
    setPage(1);
    setSelectedGame(null);
  }, []);

  const handleCriteriaChange = useCallback((next) => {
    setCriteria(next);
    setPage(1);
  }, []);

  const filtered = useMemo(() => (games ? filterGames(games, criteria) : []), [games, criteria]);
  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const startIndex = (page - 1) * PAGE_SIZE;
  const pageGames = filtered.slice(startIndex, startIndex + PAGE_SIZE);

  if (selectedGame) {
    return (
      <div className="app">
        <GameView game={selectedGame} onBack={() => setSelectedGame(null)} />
      </div>
    );
  }

  return (
    <div className="app">
      <h1>Tron Game Viewer</h1>
      {!games ? (
        <>
          <p className="hint">Drop a games.json log to get started.</p>
          <DropZone onLoad={handleLoad} />
        </>
      ) : (
        <>
          <p className="hint">
            Loaded <b>{fileName}</b> &middot; {games.length} games, {filtered.length} matching filters
            &middot; click a row to step through it, arrow keys navigate once inside, Escape returns here.
            {"  "}
            <span className="reload-link" onClick={() => setGames(null)}>
              Load a different file
            </span>
          </p>
          {meta && (
            <p className="run-meta">
              {meta.config_name && <>Config <b>{meta.config_name}</b> &middot; </>}
              Reward <b>{meta.reward_fn}</b>
              {meta.scoring_fns?.length > 0 && <> &middot; Scoring: <b>{meta.scoring_fns.join(", ")}</b></>}
              {meta.horizon != null && <> &middot; Horizon <b>{meta.horizon}</b></>}
              {meta.on_the_play != null && <> &middot; {meta.on_the_play ? "On the play" : "On the draw"}</>}
              {meta.seed != null && <> &middot; Seed <b>{meta.seed}</b></>}
            </p>
          )}
          <SearchFilters criteria={criteria} onChange={handleCriteriaChange} />
          <BatchList games={pageGames} startIndex={startIndex} onSelect={setSelectedGame} />
          <Pagination page={page} pageCount={pageCount} onPageChange={setPage} />
        </>
      )}
    </div>
  );
}
