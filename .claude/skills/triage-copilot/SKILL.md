---
name: triage-copilot
description: Fetch and classify Copilot's review suggestions on a PR as incorporate, ignore, or discuss.
disable-model-invocation: true
---

Triage GitHub Copilot's review suggestions on a pull request.

## Step 1: Resolve the PR

The PR number is: $ARGUMENTS

If no PR number was provided, detect it from the current branch:
```
gh pr view --json number --jq .number
```

## Step 2: Fetch Copilot review data

Run these commands to gather Copilot's feedback:

1. **Overview review** (from `copilot-pull-request-reviewer[bot]`):
   ```
   gh api repos/bkelly-lab/jkp-data/pulls/<PR>/reviews --jq '.[] | select(.user.login == "copilot-pull-request-reviewer[bot]") | {state, body}'
   ```

2. **Inline comments** (from user `Copilot`):
   ```
   gh api repos/bkelly-lab/jkp-data/pulls/<PR>/comments --jq '.[] | select(.user.login == "Copilot") | {path, line, body}'
   ```

3. **PR diff** for context:
   ```
   gh pr diff <PR>
   ```

If no Copilot review exists, report that and suggest running `gh pr edit <PR> --add-reviewer copilot` to request one.

## Step 3: Delegate to agent

Pass the Copilot review data and PR diff to @copilot-triage for classification.

## Step 4: Present results

Show the triage report to the reviewer. Highlight any suggestions classified as "Incorporate" or "Discuss" — these need attention.
