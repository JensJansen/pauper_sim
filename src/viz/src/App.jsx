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
  const [fileName, setFileName] = useState(null);
  const [criteria, setCriteria] = useState({});
  const [page, setPage] = useState(1);
  const [selectedGame, setSelectedGame] = useState(null);

  const handleLoad = useCallback((loadedGames, name) => {
    setGames(loadedGames);
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
          <SearchFilters criteria={criteria} onChange={handleCriteriaChange} />
          <BatchList games={pageGames} startIndex={startIndex} onSelect={setSelectedGame} />
          <Pagination page={page} pageCount={pageCount} onPageChange={setPage} />
        </>
      )}
    </div>
  );
}
