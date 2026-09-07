# Contributing

## Branching

`main` is protected and always releasable. It cannot be pushed to directly,
force-pushed, or deleted — every change arrives through a pull request whose CI
has passed.

```bash
git switch main && git pull
git switch -c feature/<short-name>
# ... work, commit ...
git push -u origin feature/<short-name>
gh pr create --fill          # or open the PR in the browser
```

Prefixes: `feature/` for new capability, `fix/` for corrections, `exp/` for
modelling experiments that may never merge, `docs/` for documentation-only work.

PRs are **squash merged**, so the PR title becomes the commit message on `main` —
write it as one. The head branch is deleted automatically on merge.

Approvals are not currently required, since the project has a single maintainer.
Raise `required_approving_review_count` to 1 in
[.github/setup-branch-protection.sh](.github/setup-branch-protection.sh) and
re-run it as soon as a second person joins.

### Protection setup

Applied once per repo, by a maintainer with `admin` scope:

```bash
gh auth login
./.github/setup-branch-protection.sh
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
