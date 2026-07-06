# Creating the CSL2 GitHub repository

This folder is a complete, push-ready repository (sibling of CSL). Run
`bootstrap_repo.sh` from inside it, or follow the manual steps.

## Option A — GitHub CLI
```bash
cd CSL2
./bootstrap_repo.sh
gh repo create CSL2 --public --source=. --remote=origin --push
```

## Option B — manual
1. Create an empty **CSL2** repository on GitHub (no README/license).
2. From inside this folder:
```bash
cd CSL2
./bootstrap_repo.sh
git remote add origin https://github.com/<owner>/CSL2.git
git push -u origin main
```

## After the first push
- Replace every `<owner>` placeholder (README, CITATION.cff, pyproject.toml, and
  the manuscript's code-availability line) with your GitHub username/org.
- Complete the author `family-names` in `CITATION.cff`.
- Confirm CI (`.github/workflows/ci.yml`) goes green (pytest + preview smoke).
- Tag and archive for a DOI: `git tag v1.0.0 && git push --tags`.

Simulation outputs (`results/`), caches, and LaTeX build artifacts are
`.gitignore`d; everything regenerates via `make reproduce`.
