# Contributing to CAPSTONE_PROJECT_V2

## Before You Start
1. **Setup your environment** - Follow the README setup steps completely
2. **Sync with main** - Always pull latest changes before starting work:
   ```bash
   git checkout main
   git pull origin main
   ```

## Workflow for Making Changes

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
# or for bug fixes:
git checkout -b bugfix/issue-description
```

### 2. Make Your Changes
- Work on your feature
- Test your changes locally
- Commit frequently with clear messages:
  ```bash
  git add .
  git commit -m "Clear description of what you changed"
  ```

### 3. Push Your Branch
```bash
git push origin feature/your-feature-name
```

### 4. Create a Pull Request
- Go to GitHub and create a PR from your branch to `main`
- Add a clear description of your changes
- Reference any related issues
- Request review from project maintainers

## Important Rules
- ✅ **DO** test your changes before committing
- ✅ **DO** keep commits organized and meaningful
- ✅ **DO** update .env.example if adding new environment variables
- ✅ **DO** document your changes in comments if needed
- ❌ **DON'T** commit `.env` files with real API keys
- ❌ **DON'T** commit large data files or model files
- ❌ **DON'T** force push to main

## Troubleshooting

### "Your branch is behind..."
```bash
git pull origin main
```

### "Untracked files" before committing
```bash
git status  # See what's untracked
# Make sure it's added to .gitignore or intentionally commit it
```

### Need to undo commits
```bash
git revert HEAD  # Safely undo last commit (creates new commit)
# OR
git reset --soft HEAD~1  # Undo last commit, keep changes staged
```

## Communication
- Update the maintainer if you're working on something significant
- Use clear branch names so others know what you're working on
- Comment on PRs to explain your reasoning
