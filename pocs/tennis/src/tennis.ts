export type PlayerId = "A" | "B";

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

export interface MatchState {
  sets: SetScore[]; // completed sets
  currentSet: CurrentSetState;
  statsA: PlayerStats;
  statsB: PlayerStats;
  history: PointEvent[];
  isEnded: boolean;
}

export function createInitialMatchState(): MatchState {
  return {
    sets: [],
    currentSet: {
      gamesA: 0,
      gamesB: 0,
      game: { type: "normal", pointsA: 0, pointsB: 0 },
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
  return gamesA === 6 && gamesB === 6;
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
  // Win set at 6 with a 2-game margin, or at 7 after a tie-break
  if ((gamesA >= 6 || gamesB >= 6) && Math.abs(gamesA - gamesB) >= 2) {
    return gamesA > gamesB ? "A" : "B";
  }
  if (gamesA === 7 || gamesB === 7) {
    return gamesA > gamesB ? "A" : "B";
  }
  return null;
}

export function formatNormalGamePoints(pointsA: number, pointsB: number): {
  displayA: string;
  displayB: string;
} {
  const mapping = ["0", "15", "30", "40"] as const;
  if (pointsA >= 3 && pointsB >= 3) {
    if (pointsA === pointsB) return { displayA: "40", displayB: "40" };
    if (pointsA === pointsB + 1) return { displayA: "Ad", displayB: "-" };
    if (pointsB === pointsA + 1) return { displayA: "-", displayB: "Ad" };
  }
  return {
    displayA: mapping[Math.min(pointsA, 3)],
    displayB: mapping[Math.min(pointsB, 3)],
  };
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
        game = { type: "normal", pointsA: 0, pointsB: 0 };
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
          game: { type: "normal", pointsA: 0, pointsB: 0 },
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
        game: { type: "normal", pointsA: 0, pointsB: 0 },
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

export function resetMatch(): MatchState {
  return createInitialMatchState();
}

export function undoLastPoint(prev: MatchState): MatchState {
  if (prev.history.length === 0) return prev;
  const newHistory = prev.history.slice(0, -1);
  let rebuilt = createInitialMatchState();
  for (const evt of newHistory) {
    rebuilt = addPoint(rebuilt, evt.winner, evt.mode);
  }
  // After undo, ensure match is not marked as ended
  return { ...rebuilt, isEnded: false };
}


