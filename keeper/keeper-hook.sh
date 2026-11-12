#!/bin/sh
# keeper-hook.sh: run `keeper hook` and fail closed.
#
# Claude Code and dsh only block on exit code 2; any other failure would let the step
# through. This wrapper turns every crash (Keeper missing, Python broken, unexpected
# exit) into exit 2 with an alert, so a broken guard never means an open door.
#
# KEEPER_BIN can point at the keeper command (default: keeper on PATH).

"${KEEPER_BIN:-keeper}" hook "$@"
code=$?
if [ "$code" -eq 0 ] || [ "$code" -eq 2 ]; then
    exit "$code"
fi
cat >&2 <<MSG
ALERT: The Keeper itself failed (exit code $code), so this step is blocked.

What happened, step by step:
1. The hook ran: keeper hook $*
2. It ended with exit code $code instead of a normal answer.
3. Stopped. Nothing was retried or worked around. Waiting for Al.

In simple terms: the guard broke, so the door stays shut until Al looks.
MSG
exit 2
