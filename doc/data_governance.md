# Data governance

> Confirm the specifics with the data owner and information governance lead before
> any data is received. This file records what was agreed.

## Classification

Site-level energy consumption data is **not** patient data, but it is commercially
sensitive and can be security-sensitive (occupancy patterns, critical
infrastructure). Treat it as confidential.

| Item | Status |
|---|---|
| Data sharing agreement in place | TBD |
| Data owner | TBD |
| IG / DPO sign-off | TBD |
| Retention period | TBD |
| Permitted use | TBD |
| Onward sharing permitted | TBD |

## Rules in this repository

1. **No data is committed to git.** `code/data/` is git-ignored end to end; the
   `check-added-large-files` and `detect-private-key` pre-commit hooks are a
   backstop, not the control.
2. **Notebook outputs are stripped on commit** (`nbstripout`) — outputs can embed
   site-level values.
3. **No credentials in code.** `.env` is git-ignored; read secrets via
   `sense_energy.config.SECRETS`.
4. **Aggregate before publishing.** Figures and reports leaving the project should
   be anonymised or aggregated unless the data owner has approved site-level
   disclosure.
5. **If data is committed by accident**, treat it as a disclosure incident: rotate
   any exposed credentials, notify the data owner, and rewrite history — deleting
   the file in a later commit does not remove it.

## Storage

| Location | Permitted contents |
|---|---|
| `code/data/` (local) | Full source data |
| Git remote | Code and docs only |
| Shared drives / cloud | Per the data sharing agreement — TBD |
