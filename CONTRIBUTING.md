# Contributing

## Branching

Development happens on a working branch (currently `feature/nhs-data-cleaning`).
`main` holds what has been shown to work: when a piece is final, merge the
branch into `main`. No branch protection is configured; CI runs on pushes and
pull requests as a check, not a gate.

```bash
git switch main && git merge --no-ff feature/nhs-data-cleaning && git push
```

## Commits

Conventional-commit style prefixes: `feat:`, `fix:`, `data:`, `docs:`, `refactor:`,
`test:`, `chore:`.

## Before you push

```bash
make format
make lint
make test
```

`pre-commit` runs these hooks automatically (`make setup` installs them). It also
strips notebook outputs — this is deliberate, since outputs can contain site-level data.

## Rules that matter here

1. **Never commit data.** Not even a "small sample". `code/data/` is git-ignored;
   don't force-add anything through it.
2. **No secrets in code.** Use `.env` (git-ignored) and read via `sense_energy.config`.
3. **Notebooks are scratch.** Anything reused belongs in `code/src/sense_energy/`.
4. **New data source?** Document it in `doc/data_dictionary.md` in the same PR.
5. **Changed the modelling approach?** Write an ADR in `doc/adr/`.
