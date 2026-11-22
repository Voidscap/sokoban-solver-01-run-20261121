# Sokoban solver

A push-optimal Sokoban solver for make.fun's `sokoban-solver-01-run-20261121`
competition. One run answers all 5 instances of a set.

## How it works

A generated level is a hallway with 1–4 rooms off it, 3–4 boxes per room,
and a box pushed out of its room can never come back — so the level's
push-optimal count is the sum of its rooms' optima. The solver exploits
that structure:

1. **Room decomposition.** The dead squares of the player's row (the
   hallway, where a box reaches no goal) are cut out of the floor; each
   remaining part holding boxes is solved as its own room, the other
   rooms' boxes standing as walls (on their goals once their room is
   done), the player walking the hallway between rooms.
2. **Push-optimal A\* per room.** States are the boxes plus the player's
   reachable region; each step is one push. The heuristic is the cheapest
   assignment of boxes to goals by push-distance (a reverse search from
   each goal that never overestimates), so the first solution found uses
   the fewest pushes. Pruning removes only states no solution can pass
   through: boxes on dead squares and frozen 2×2 blocks of walls and
   boxes.
3. **Deterministic budgets.** Every search is capped by expanded states,
   never by the wall clock, so two runs write the same bytes. No
   randomness, no clock reads, no network.

The whole-level fallback (used only when a level does not split into
rooms) runs the same A\* over the full board, with the Hungarian method
for the assignment when there are too many boxes to enumerate
permutations.

## Practice score

On the kit's practice set the solver is push-optimal on every instance:

| instance | P\* | pushes | points |
|----------|-----|--------|--------|
| i001 | 13 | 13 | 1000 |
| i002 | 16 | 16 | 1000 |
| i003 | 29 | 29 | 1000 |
| i004 | 47 | 47 | 1000 |
| i005 | 60 | 60 | 1000 |

**Practice score: 1,000,000** (the house baseline, plain-astar 1.0.0,
scores 200,000 on the same set). The full set solves in about 0.3 s
locally, far inside the 60 s wall budget, and three further randomly
drawn sets from the kit's generator were also solved push-optimally on
every instance.

## Running it

```sh
node main.ts            # reads /input/<id>/level.txt, writes /output/<id>/solution.txt
```

Locally, point it at the practice set instead:

```sh
MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Grade it with the kit (from the unpacked kit directory, no Docker):

```sh
node cli/makefun.mjs grade-local --kit . --bot <this repository> --driver process
```

Requires Node 22 or later (erasable TypeScript only, no dependencies).

## License

MIT — see `LICENSE`.

Referral code: `qsepttqp`
