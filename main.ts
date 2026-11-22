// Sokoban solver: push-based A* over rooms.
//
// Strategy: a generated level is a hallway with 1-4 rooms off it, 3-4 boxes
// per room, and a box pushed out of its room can never come back, so the
// level's push-optimal count is the sum of its rooms' optima. The solver
// cuts the level into rooms, solves each room on its own with an A* search
// over pushes (states: the boxes plus the player's reachable region), and
// walks the hallway between rooms. The search is push-optimal: the
// heuristic is the cheapest assignment of boxes to goals by push-distance,
// which never overestimates, and the pruning (dead squares, frozen 2x2
// blocks) only removes states no solution can pass through.
//
// Deterministic: budgets are counted in expanded states, never in wall
// time; every tie is broken by insertion order. No clock, no randomness,
// no environment reads beyond the input/output directories.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets (states expanded), never the clock.
const ROOM_EXPAND_CAP = 400_000;
const WHOLE_EXPAND_CAP = 150_000;
const GLOBAL_EXPAND_CAP = 2_500_000;
let globalExpanded = 0;

const DIRS: [number, number, string][] = [
  [0, -1, 'u'],
  [0, 1, 'd'],
  [-1, 0, 'l'],
  [1, 0, 'r'],
];

type Level = {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
};

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// Push-distance tables: pd[g * size + x] is the fewest pushes that move a
// box from square x onto goal g when the player may stand anywhere (the
// "teleporting player" bound). It never overestimates the true push cost,
// and it is infinite exactly on the dead squares. Built by a reverse
// breadth-first search over box squares: a box at y came from y - d, which
// needed the player standing at y - 2d.
function pushDistances(level: Level): Int32Array {
  const { w, wall, goals } = level;
  const size = wall.length;
  const pd = new Int32Array(goals.length * size).fill(-1);
  const queue = new Int32Array(size);
  for (let gi = 0; gi < goals.length; gi++) {
    const g = goals[gi]!;
    const base = gi * size;
    pd[base + g] = 0;
    let qh = 0;
    let qt = 0;
    queue[qt++] = g;
    while (qh < qt) {
      const y = queue[qh++]!;
      const d0 = pd[base + y]!;
      for (let k = 0; k < 4; k++) {
        const [dx, dy] = DIRS[k]!;
        const d = dy * w + dx;
        const x = y - d; // the box's previous square
        const stand = y - 2 * d; // where the player stood to push it here
        if (x < 0 || x >= size || stand < 0 || stand >= size) continue;
        if (wall[x] || wall[stand] || pd[base + x] >= 0) continue;
        pd[base + x] = d0 + 1;
        queue[qt++] = x;
      }
    }
  }
  return pd;
}

// All permutations of [0..n-1], built once and reused for the assignment.
function permutations(n: number): number[][] {
  const out: number[][] = [];
  const a = Array.from({ length: n }, (_, i) => i);
  const gen = (k: number): void => {
    if (k === n) {
      out.push(a.slice());
      return;
    }
    for (let i = k; i < n; i++) {
      [a[k], a[i]] = [a[i]!, a[k]!];
      gen(k + 1);
      [a[k], a[i]] = [a[i]!, a[k]!];
    }
  };
  gen(0);
  return out;
}

// Min-cost assignment by the Hungarian method (O(n^3)), for box counts
// where enumerating permutations would explode. `cost` holds finite
// entries, or BIG where a box cannot reach a goal; a total of at least
// BIG means no assignment exists.
const BIG = 1_000_000;
function hungarian(cost: number[][]): number {
  const n = cost.length;
  const u = new Array<number>(n + 1).fill(0);
  const v = new Array<number>(n + 1).fill(0);
  const p = new Array<number>(n + 1).fill(0);
  const way = new Array<number>(n + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array<number>(n + 1).fill(Infinity);
    const used = new Array<boolean>(n + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0]!;
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const cur = cost[i0 - 1]![j - 1]! - u[i0]! - v[j]!;
        if (cur < minv[j]!) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j]! < delta) {
          delta = minv[j]!;
          j1 = j;
        }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) {
          const pj = p[j]!;
          u[pj] = u[pj]! + delta;
          v[j] = v[j]! - delta;
        } else {
          minv[j] = minv[j]! - delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0]!;
      p[j0] = p[j1]!;
      j0 = j1;
    } while (j0 !== 0);
  }
  let total = 0;
  for (let j = 1; j <= n; j++) total += cost[p[j]! - 1]![j - 1]!;
  return total;
}

// Cheapest assignment of boxes to goals by push-distance; Infinity when
// some box reaches no goal at all.
function assignmentHeuristic(pd: Int32Array, size: number, bs: number[], perms: number[][] | null): number {
  const n = bs.length;
  if (perms !== null) {
    let best = Infinity;
    for (const perm of perms) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const d = pd[perm[i]! * size + bs[i]!]!;
        if (d < 0) {
          sum = Infinity;
          break;
        }
        sum += d;
      }
      if (sum < best) best = sum;
    }
    return best;
  }
  const cost: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      const d = pd[j * size + bs[i]!]!;
      row.push(d < 0 ? BIG : d);
    }
    cost.push(row);
  }
  const total = hungarian(cost);
  return total >= BIG ? Infinity : total;
}
// with `stamp` and returns the lowest square reached.
function flood(
  level: Level,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const [dx, dy] = DIRS[k]!;
      const q = p + dy * w + dx;
      if (!wall[q] && !occupied[q] && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

// Whether the box just pushed to `t` sits in a frozen 2x2 block: four
// squares of walls and boxes, one of them a box off its goal. No box in
// such a block can ever move again, so the state is dead.
function frozenBlock(level: Level, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a])) continue;
    if (!(wall[b] || occupied[b])) continue;
    if (!(wall[c] || occupied[c])) continue;
    if (!(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) {
      return true;
    }
  }
  return false;
}

// The shortest walk from `from` to `to` around walls and boxes, in lower
// LURD letters. The caller guarantees `to` is reachable.
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  let s = '';
  if (prev[to] === -1) return s; // unreachable: cannot happen for a generated push, but never hang
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

function stateKey(boxes: number[], low: number, size: number): number | string {
  if (boxes.length <= 5) {
    let k = low;
    for (const b of boxes) k = k * size + b;
    return k;
  }
  return `${low}|${boxes.join(',')}`;
}

// Push-optimal A* over pushes for one (sub-)level. Returns the pushes and
// the moves, or null when the state cap runs out or no solution exists.
function solvePushes(level: Level, expandCap: number): { pushes: number; moves: string } | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const pd = pushDistances(level);
  const n = level.boxes.length;
  // Permutations up to 5 boxes; the Hungarian method above that, so the
  // whole-level fallback with many boxes cannot blow up factorially.
  const perms = n <= 5 ? permutations(n) : null;
  const solved = (bs: number[]): boolean => bs.every((b) => goal[b]);
  if (level.boxes.some((b) => goals.every((_, gi) => pd[gi * size + b]! < 0))) return null;

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  // Cheapest assignment of boxes to goals by push-distance; Infinity when
  // some box reaches no goal at all.
  const heuristic = (bs: number[]): number => assignmentHeuristic(pd, size, bs, perms);

  const boxesOf: number[][] = [level.boxes.slice()];
  const playerOf: number[] = [level.player];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushTo: number[] = [-1];
  const pushDir: number[] = [-1];
  const gScore: number[] = [0];
  const best = new Map<number | string, number>();
  for (const b of level.boxes) occupied[b] = 1;
  const startLow = flood(level, occupied, level.player, seen, ++stamp, stack);
  for (const b of level.boxes) occupied[b] = 0;
  best.set(stateKey(level.boxes, startLow, size), 0);

  // Open list: binary heap on (f, -g, insertion order).
  const heap: [number, number, number, number][] = [];
  const less = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  const h0 = heuristic(level.boxes);
  if (h0 === Infinity) return null;
  heapPush([h0, 0, 0, 0]);
  let seq = 1;
  let expanded = 0;
  let found = solved(level.boxes) ? 0 : -1;

  while (heap.length > 0 && found < 0) {
    const [, g, , s] = heapPop();
    if (gScore[s] !== g) continue;
    if (expanded >= expandCap || globalExpanded >= GLOBAL_EXPAND_CAP) return null;
    expanded++;
    globalExpanded++;
    const boxes = boxesOf[s]!;
    for (const b of boxes) occupied[b] = 1;
    const here = ++stamp;
    flood(level, occupied, playerOf[s]!, seen, here, stack);
    for (let bi = 0; bi < n && found < 0; bi++) {
      const b = boxes[bi]!;
      for (let k = 0; k < 4; k++) {
        const [dx, dy] = DIRS[k]!;
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t]) continue;
        if (goals.every((_, gi) => pd[gi * size + t]! < 0)) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(level, occupied, goal, t);
        let key: number | string = '';
        let next: number[] | null = null;
        let low = -1;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          low = flood(level, occupied, b, mark, ++stamp, stack);
          key = stateKey(next, low, size);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || next === null) continue;
        const seenG = best.get(key);
        if (seenG !== undefined && seenG <= g + 1) continue;
        const h = heuristic(next);
        if (h === Infinity) continue;
        best.set(key, g + 1);
        const idx = boxesOf.length;
        boxesOf.push(next);
        playerOf.push(b);
        parent.push(s);
        pushFrom.push(b);
        pushTo.push(t);
        pushDir.push(k);
        gScore.push(g + 1);
        heapPush([g + 1 + h, g + 1, seq++, idx]);
        if (solved(next)) {
          found = idx;
          break;
        }
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes.slice();
  for (const s of path) {
    const from = pushFrom[s]!;
    const to = pushTo[s]!;
    const d = to - from;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += DIRS[pushDir[s]!]![2];
    player = from;
    boxes = boxesOf[s]!.slice();
  }
  return { pushes: path.length, moves };
}

type Room = { squares: number[]; boxes: number[]; goals: number[] };

// Split the level into rooms: cut the dead squares of the player's row
// (the hallway, where a box reaches no goal) out of the floor; each
// remaining part holding boxes is a room. Null when the level is no set
// of rooms (a box in the hallway, a goal outside any room, or a room whose
// boxes and goals do not match).
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const pd = pushDistances(level);
  const size = wall.length;
  const row = Math.floor(player / w);
  const cut = new Uint8Array(size);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (!wall[i] && goals.every((_, gi) => pd[gi * size + i]! < 0)) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(size).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < size; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0)) return null;
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

// Solve one level room by room: each room's push-optimal solution in turn,
// the other rooms' boxes standing as walls (on their goals once their room
// is done), the player walking on from where the last room left it.
function solveLevel(text: string): string {
  const level = parseLevel(text);
  const rooms = roomsOf(level);
  if (rooms === null) {
    const whole = solvePushes(level, WHOLE_EXPAND_CAP);
    return whole === null ? '' : whole.moves;
  }
  let player = level.player;
  let moves = '';
  const done = new Set<number>();
  while (done.size < rooms.length) {
    let progress = false;
    for (let ri = 0; ri < rooms.length; ri++) {
      if (done.has(ri)) continue;
      const room = rooms[ri]!;
      const wall = level.wall.slice();
      for (let oj = 0; oj < rooms.length; oj++) {
        if (oj === ri) continue;
        const other = rooms[oj]!;
        for (const b of done.has(oj) ? other.goals : other.boxes) wall[b] = 1;
      }
      const sub: Level = {
        w: level.w,
        h: level.h,
        wall,
        goals: room.goals.slice().sort((a, b) => a - b),
        boxes: room.boxes.slice().sort((a, b) => a - b),
        player,
      };
      const solved = solvePushes(sub, ROOM_EXPAND_CAP);
      if (solved === null) continue;
      moves += solved.moves;
      player = endOf(level.w, player, solved.moves);
      done.add(ri);
      progress = true;
      break;
    }
    if (!progress) return '';
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
