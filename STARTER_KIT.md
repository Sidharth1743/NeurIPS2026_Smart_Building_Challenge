# The simulator we are measured against

`starter-kit/` is the organiser's Python simulator, unmodified. It is not vendored in this
repository because it is a public clone of the same commit.

- upstream: https://github.com/smart-buildings-2026/starter-kit
- commit: `23a4560df0e2fe61054dd21066fa0c4b5da2c79c`
- licence: Apache-2.0, see the `LICENSE` in the clone

`./run.sh setup` clones it to `starter-kit/` at that commit and creates the Python 3.12
environment the oracle needs.

It is the oracle, not our code. Nothing in `complete_port/` writes to it.
