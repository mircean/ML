import { useMemo, useState } from 'react'
import './App.css'
import {
  addPoint,
  createInitialMatchState,
  endMatch,
  formatNormalGamePoints,
  undoLastPoint,
  resetMatch,
} from './tennis'
import type { MatchState, PointMode } from './tennis'

function App() {
  const [state, setState] = useState<MatchState>(createInitialMatchState())
  

  const scoreDisplay = useMemo(() => {
    const { game, gamesA, gamesB } = state.currentSet
    if (game.type === 'normal') {
      const { displayA, displayB } = formatNormalGamePoints(
        game.pointsA,
        game.pointsB,
      )
      return {
        set: `${gamesA} - ${gamesB}`,
        game: `${displayA} : ${displayB}`,
        subtitle: 'Game',
      }
    }
    return {
      set: `${gamesA} - ${gamesB}`,
      game: `${game.pointsA} : ${game.pointsB}`,
      subtitle: 'Tie-break',
    }
  }, [state])

  const handleAction = (player: 'A' | 'B', action: PointMode) => {
    if (state.isEnded) return
    const winner: 'A' | 'B' =
      action === 'ace' || action === 'winner' || action === 'other-winner'
        ? player
        : player === 'A'
        ? 'B'
        : 'A'
    setState((s) => addPoint(s, winner, action))
  }

  return (
    <div className="container">
      <h1>Tennis Match Stats</h1>

      <div className="scoreboard">
        <div className="score-row">
          <div className="score-label">Set</div>
          <div className="score-value">{scoreDisplay.set}</div>
        </div>
        <div className="score-row">
          <div className="score-label">{scoreDisplay.subtitle}</div>
          <div className="score-value game">{scoreDisplay.game}</div>
        </div>
      </div>

      <div className="controls">
        <div className="players">
          <div className="player-section">
            <div className="player-title">Player A</div>
            <div className="actions-grid">
              <button className="action-win" disabled={state.isEnded} onClick={() => handleAction('A', 'ace')}>Ace</button>
              <button className="action-win" disabled={state.isEnded} onClick={() => handleAction('A', 'winner')}>Winner</button>
              <button className="action-win" disabled={state.isEnded} onClick={() => handleAction('A', 'other-winner')}>Other winner</button>
              <button className="action-err" disabled={state.isEnded} onClick={() => handleAction('A', 'double-fault')}>Double fault</button>
              <button className="action-err" disabled={state.isEnded} onClick={() => handleAction('A', 'forced-error')}>Forced error</button>
              <button className="action-err" disabled={state.isEnded} onClick={() => handleAction('A', 'unforced-error')}>Unforced error</button>
            </div>
          </div>
          <div className="player-section">
            <div className="player-title">Player B</div>
            <div className="actions-grid">
              <button className="action-win" disabled={state.isEnded} onClick={() => handleAction('B', 'ace')}>Ace</button>
              <button className="action-win" disabled={state.isEnded} onClick={() => handleAction('B', 'winner')}>Winner</button>
              <button className="action-win" disabled={state.isEnded} onClick={() => handleAction('B', 'other-winner')}>Other winner</button>
              <button className="action-err" disabled={state.isEnded} onClick={() => handleAction('B', 'double-fault')}>Double fault</button>
              <button className="action-err" disabled={state.isEnded} onClick={() => handleAction('B', 'forced-error')}>Forced error</button>
              <button className="action-err" disabled={state.isEnded} onClick={() => handleAction('B', 'unforced-error')}>Unforced error</button>
            </div>
          </div>
        </div>
        <div className="match-buttons">
          <button onClick={() => setState((s) => endMatch(s))} disabled={state.isEnded}>
            End Match
          </button>
          <button onClick={() => setState((s) => undoLastPoint(s))} disabled={state.history.length === 0}>Undo</button>
          <button onClick={() => setState(resetMatch())}>Reset</button>
        </div>
      </div>

      <div className="stats">
        <h2>Statistics</h2>
        <div className="stats-grid">
          <div className="stats-col">
            <div className="player">Player A</div>
            <div className="stat"><span>Aces</span><b>{state.statsA.aces}</b></div>
            <div className="stat"><span>Winner</span><b>{state.statsA.winners}</b></div>
            <div className="stat"><span>Double Faults</span><b>{state.statsA.doubleFaults}</b></div>
            <div className="stat"><span>Forced Errors</span><b>{state.statsA.forcedErrors}</b></div>
            <div className="stat"><span>Unforced Errors</span><b>{state.statsA.unforcedErrors}</b></div>
          </div>
          <div className="stats-col">
            <div className="player">Player B</div>
            <div className="stat"><span>Aces</span><b>{state.statsB.aces}</b></div>
            <div className="stat"><span>Winner</span><b>{state.statsB.winners}</b></div>
            <div className="stat"><span>Double Faults</span><b>{state.statsB.doubleFaults}</b></div>
            <div className="stat"><span>Forced Errors</span><b>{state.statsB.forcedErrors}</b></div>
            <div className="stat"><span>Unforced Errors</span><b>{state.statsB.unforcedErrors}</b></div>
          </div>
        </div>

        {state.sets.length > 0 && (
          <div className="completed-sets">
            <h3>Completed Sets</h3>
            <ul>
              {state.sets.map((s, idx) => (
                <li key={idx}>
                  Set {idx + 1}: {s.gamesA} - {s.gamesB}
                  {typeof s.tieBreakA === 'number' && typeof s.tieBreakB === 'number' ? (
                    <span> (TB {s.tieBreakA}:{s.tieBreakB})</span>
                  ) : null}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
