# Lessons

- 2026-05-25: For Chinese-Standard-Mahjong/MCR policy code, never allow `HU` from a complete-hand check alone. `HU` is legal only when the official fan requirement is satisfied. If the official fan checker is unavailable in a runtime artifact, suppress `HU` rather than risking `WH` penalties.
- 2026-05-25: Treat unknown fan as not enough fan. The bot must wait instead of emitting `HU` unless the full model's official checker or the source package's conservative lower-bound gate proves at least 8 fan.
