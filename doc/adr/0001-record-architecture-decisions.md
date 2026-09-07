# ADR 0001: Record architecture decisions

- **Status:** Accepted
- **Date:** 2026-09-07

## Context

Modelling projects accumulate decisions — the validation scheme, the target
variable, which sites to exclude — that are obvious when made and mysterious six
months later. Reconstructing them from the git log is expensive.

## Decision

Record significant decisions as short ADRs in `doc/adr/`, numbered sequentially,
using this template. A decision is significant if reversing it would mean redoing
work, or if a newcomer would otherwise ask "why is it done this way?".

## Consequences

- One extra file per significant decision.
- PRs that change the modelling approach are expected to include an ADR.
- Superseded ADRs are kept and marked `Superseded by ADR-NNNN`, not deleted.
