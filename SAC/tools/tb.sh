#!/bin/bash
# Alias: one board for every SAC run on :6007 (see tb_view.sh).
#   SAC/tools/tb.sh            -> http://localhost:6007  (all runs)
#   SAC/tools/tb.sh --watch    -> keep refreshing the run links every 60 s
exec "$(cd "$(dirname "$0")" && pwd)/tb_view.sh" "$@"
