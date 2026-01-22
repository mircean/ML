export type PlayerId = "A" | "B";

export type ScoringSystem = "normal" | "simple";

export type PointMode =
  | "ace"
  | "winner"
  | "other-winner"
  | "double-fault"
  | "forced-error"
  | "unforced-error";

export interface PointEvent {
  winner: PlayerId;
  mode: PointMode;
  timestamp: number;
}

export interface PlayerStats {
  aces: number;
  winners: number;
  forcedErrors: number;
  unforcedErrors: number;
  doubleFaults: number;
}

export interface SetScore {
  gamesA: number;
  gamesB: number;
  tieBreakA?: number;
  tieBreakB?: number;
}

export type GameType = "normal" | "tiebreak";

export interface NormalGame {
  type: "normal";
  pointsA: number; // raw points count within game
  pointsB: number;
}

export interface TieBreakGame {
  type: "tiebreak";
  pointsA: number; // tie-break points (race to 7, win by 2)
  pointsB: number;
}

export type CurrentGame = NormalGame | TieBreakGame;

export interface CurrentSetState {
  gamesA: number;
  gamesB: number;
  game: CurrentGame;
}

export interface SimpleGameState {
  pointsA: number; // Player A starts at -2
  pointsB: number; // Player B starts at 0
  gamesA: number;  // Games won by A in current set
  gamesB: number;  // Games won by B in current set
  setsA: number;   // Sets won by A
  setsB: number;   // Sets won by B
}

export interface MatchState {
  scoringSystem: ScoringSystem;
  sets: SetScore[]; // completed sets (only for normal scoring)
  currentSet: CurrentSetState; // only for normal scoring
  simpleGame: SimpleGameState; // only for simple scoring
  statsA: PlayerStats;
  statsB: PlayerStats;
  history: PointEvent[];
  isEnded: boolean;
}

export function createInitialMatchState(scoringSystem: ScoringSystem = "normal"): MatchState {
  return {
    scoringSystem,
    sets: [],
    currentSet: {
      gamesA: 0,
      gamesB: 0,
      game: { type: "normal", pointsA: -2, pointsB: 0 },
    },
    simpleGame: {
      pointsA: -2,
      pointsB: 0,
      gamesA: 0,
      gamesB: 0,
      setsA: 0,
      setsB: 0,
    },
    statsA: {
      aces: 0,
      winners: 0,
      forcedErrors: 0,
      unforcedErrors: 0,
      doubleFaults: 0,
    },
    statsB: {
      aces: 0,
      winners: 0,
      forcedErrors: 0,
      unforcedErrors: 0,
      doubleFaults: 0,
    },
    history: [],
    isEnded: false,
  };
}

function incrementStats(stats: PlayerStats, mode: PointMode): PlayerStats {
  if (mode === "ace") return { ...stats, aces: stats.aces + 1 };
  if (mode === "winner") return { ...stats, winners: stats.winners + 1 };
  if (mode === "double-fault")
    return { ...stats, doubleFaults: stats.doubleFaults + 1 };
  if (mode === "forced-error")
    return { ...stats, forcedErrors: stats.forcedErrors + 1 };
  return { ...stats, unforcedErrors: stats.unforcedErrors + 1 };
}

function isSetInTieBreak(gamesA: number, gamesB: number): boolean {
  // No tiebreaks with first-to-4 sets
  return false;
}

function hasNormalGameWinner(pointsA: number, pointsB: number): PlayerId | null {
  if (pointsA >= 4 || pointsB >= 4) {
    const lead = Math.abs(pointsA - pointsB);
    if (lead >= 2) {
      return pointsA > pointsB ? "A" : "B";
    }
  }
  return null;
}

function hasTieBreakWinner(pointsA: number, pointsB: number): PlayerId | null {
  if (pointsA >= 7 || pointsB >= 7) {
    const lead = Math.abs(pointsA - pointsB);
    if (lead >= 2) {
      return pointsA > pointsB ? "A" : "B";
    }
  }
  return null;
}

function hasSetWinner(gamesA: number, gamesB: number): PlayerId | null {
  // First to 4 games wins the set, 4-3 is OK
  if (gamesA >= 4) return "A";
  if (gamesB >= 4) return "B";
  return null;
}

export function formatNormalGamePoints(pointsA: number, pointsB: number): {
  displayA: string;
  displayB: string;
} {
  // Player A starts at -2 (-30), mapping: -2→-30, -1→-15, 0→0, 1→15, 2→30, 3→40
  const mappingA: Record<number, string> = {
    [-2]: "-30", [-1]: "-15", 0: "0", 1: "15", 2: "30", 3: "40"
  };
  const mappingB = ["0", "15", "30", "40"] as const;

  // Deuce and advantage handling (both at 3+ effective points, i.e., A at 3+, B at 3+)
  if (pointsA >= 3 && pointsB >= 3) {
    if (pointsA === pointsB) return { displayA: "40", displayB: "40" };
    if (pointsA === pointsB + 1) return { displayA: "Ad", displayB: "-" };
    if (pointsB === pointsA + 1) return { displayA: "-", displayB: "Ad" };
  }

  const displayA = pointsA >= 3 ? "40" : (mappingA[pointsA] ?? String(pointsA));
  const displayB = mappingB[Math.min(pointsB, 3)];

  return { displayA, displayB };
}

function hasSimpleGameWinner(pointsA: number, pointsB: number): PlayerId | null {
  // First to 5 with diff of 2 wins the game
  if (pointsA >= 5 && pointsA - pointsB >= 2) return "A";
  if (pointsB >= 5 && pointsB - pointsA >= 2) return "B";
  return null;
}

function hasSimpleSetWinner(gamesA: number, gamesB: number): PlayerId | null {
  // First to 4 games wins the set
  if (gamesA >= 4) return "A";
  if (gamesB >= 4) return "B";
  return null;
}

export function addPoint(
  prev: MatchState,
  winner: PlayerId,
  mode: PointMode
): MatchState {
  if (prev.isEnded) return prev;

  const now = Date.now();
  const history: PointEvent[] = [
    ...prev.history,
    { winner, mode, timestamp: now },
  ];

  // For aces/winners: credit winner. For errors: credit the error maker (loser).
  let statsA = prev.statsA;
  let statsB = prev.statsB;
  if (mode === "other-winner") {
    // winner gets point but no stat increment
  } else if (mode === "ace" || mode === "winner") {
    if (winner === "A") statsA = incrementStats(statsA, mode);
    else statsB = incrementStats(statsB, mode);
  } else {
    const errorMaker: PlayerId = winner === "A" ? "B" : "A";
    if (errorMaker === "A") statsA = incrementStats(statsA, mode);
    else statsB = incrementStats(statsB, mode);
  }

  // Simple scoring system
  if (prev.scoringSystem === "simple") {
    const pointsA = prev.simpleGame.pointsA + (winner === "A" ? 1 : 0);
    const pointsB = prev.simpleGame.pointsB + (winner === "B" ? 1 : 0);
    const gameWinner = hasSimpleGameWinner(pointsA, pointsB);

    if (gameWinner) {
      // Game won, update games count and reset points
      const gamesA = prev.simpleGame.gamesA + (gameWinner === "A" ? 1 : 0);
      const gamesB = prev.simpleGame.gamesB + (gameWinner === "B" ? 1 : 0);
      const setWinner = hasSimpleSetWinner(gamesA, gamesB);

      if (setWinner) {
        // Set won, update sets count and reset games
        return {
          ...prev,
          simpleGame: {
            pointsA: -2,
            pointsB: 0,
            gamesA: 0,
            gamesB: 0,
            setsA: prev.simpleGame.setsA + (setWinner === "A" ? 1 : 0),
            setsB: prev.simpleGame.setsB + (setWinner === "B" ? 1 : 0),
          },
          statsA,
          statsB,
          history,
        };
      }

      return {
        ...prev,
        simpleGame: {
          ...prev.simpleGame,
          pointsA: -2,
          pointsB: 0,
          gamesA,
          gamesB,
        },
        statsA,
        statsB,
        history,
      };
    }

    return {
      ...prev,
      simpleGame: { ...prev.simpleGame, pointsA, pointsB },
      statsA,
      statsB,
      history,
    };
  }

  // Normal scoring system
  let { gamesA, gamesB, game } = prev.currentSet;

  if (game.type === "normal") {
    const pointsA = game.pointsA + (winner === "A" ? 1 : 0);
    const pointsB = game.pointsB + (winner === "B" ? 1 : 0);
    const gameWinner = hasNormalGameWinner(pointsA, pointsB);

    if (gameWinner) {
      if (gameWinner === "A") gamesA += 1;
      else gamesB += 1;

      // Determine next game type (tie-break at 6-6)
      if (isSetInTieBreak(gamesA, gamesB)) {
        game = { type: "tiebreak", pointsA: 0, pointsB: 0 };
      } else {
        game = { type: "normal", pointsA: -2, pointsB: 0 };
      }
    } else {
      game = { type: "normal", pointsA, pointsB };
    }
  } else {
    // tie-break game
    const pointsA = game.pointsA + (winner === "A" ? 1 : 0);
    const pointsB = game.pointsB + (winner === "B" ? 1 : 0);
    const tbWinner = hasTieBreakWinner(pointsA, pointsB);
    if (tbWinner) {
      if (tbWinner === "A") {
        gamesA = 7;
      } else {
        gamesB = 7;
      }
      // Record completed set including tie-break points
      const completedSet: SetScore = {
        gamesA,
        gamesB,
        tieBreakA: pointsA,
        tieBreakB: pointsB,
      };
      return {
        ...prev,
        sets: [...prev.sets, completedSet],
        currentSet: {
          gamesA: 0,
          gamesB: 0,
          game: { type: "normal", pointsA: -2, pointsB: 0 },
        },
        statsA,
        statsB,
        history,
      };
    } else {
      game = { type: "tiebreak", pointsA, pointsB };
    }
  }

  // Check for set winner in non tie-break flow
  const setWinner = hasSetWinner(gamesA, gamesB);
  if (setWinner && game.type !== "tiebreak") {
    const completedSet: SetScore = { gamesA, gamesB };
    return {
      ...prev,
      sets: [...prev.sets, completedSet],
      currentSet: {
        gamesA: 0,
        gamesB: 0,
        game: { type: "normal", pointsA: -2, pointsB: 0 },
      },
      statsA,
      statsB,
      history,
    };
  }

  return {
    ...prev,
    currentSet: { gamesA, gamesB, game },
    statsA,
    statsB,
    history,
  };
}

export function endMatch(prev: MatchState): MatchState {
  return { ...prev, isEnded: true };
}

export function resetMatch(scoringSystem: ScoringSystem): MatchState {
  return createInitialMatchState(scoringSystem);
}

export function undoLastPoint(prev: MatchState): MatchState {
  if (prev.history.length === 0) return prev;
  const newHistory = prev.history.slice(0, -1);
  let rebuilt = createInitialMatchState(prev.scoringSystem);
  for (const evt of newHistory) {
    rebuilt = addPoint(rebuilt, evt.winner, evt.mode);
  }
  // After undo, ensure match is not marked as ended
  return { ...rebuilt, isEnded: false };
}


