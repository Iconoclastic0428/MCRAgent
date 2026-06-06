# Agent Operating Rules

## Tjong Paper Replication

- Treat the active paper-training run as golden progress. Do not delete, restart, replace, interrupt, or relaunch a training job once it has started unless the user explicitly instructs it or the job has objectively failed.
- Follow only the paper's training path for the Tjong replication. Do not mix in unrelated work such as E3/E4 batch checkpoint evaluation, previous non-paper experiments, or earlier thread leftovers.
- Do exactly one operational thing at a time. While a training job is running, monitor and report its status only; do not run conversion jobs, all-sources prep jobs, cleanup jobs, eval jobs, or manifest rewrites in parallel unless the user explicitly asks.
- Do not repeatedly rerun CPU conversion or all-sources prep after the required paper artifacts already exist and have been validated. Use existing validated artifacts as inputs.
- Preserve GPU hours. Before taking any action that can consume, reset, or waste GPU time, verify it is required by the paper path and confirm it will not discard current progress.
- If there is any ambiguity between paper replication and older unrelated requests, stop and ask before acting.
