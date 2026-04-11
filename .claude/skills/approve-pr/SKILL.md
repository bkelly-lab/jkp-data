---
name: approve-pr
description: Pre-flight checks, approve, squash-merge, and delete branch for a PR.
disable-model-invocation: true
---

Approve and merge a pull request after running pre-flight safety checks.

## Step 1: Resolve the PR

The PR number is: $ARGUMENTS

If no PR number was provided, detect it from the current branch:
```
gh pr view --json number --jq .number
```

## Step 2: Pre-flight checks

Run all checks before proceeding:

1. **CI status**:
   ```
   gh pr checks <PR>
   ```
   If any required checks are failing, report and stop. If checks are pending, report the status and ask whether to wait or proceed.

2. **Merge conflicts**:
   ```
   gh pr view <PR> --json mergeable --jq .mergeable
   ```
   If `CONFLICTING`, report and stop — author must resolve conflicts first.

3. **Existing reviews**:
   ```
   gh api repos/bkelly-lab/jkp-data/pulls/<PR>/reviews --jq '.[] | {user: .user.login, state: .state}'
   ```
   Show existing reviews for awareness.

4. **PR summary**:
   ```
   gh pr view <PR> --json title,author,commits --jq '{title, author: .author.login, commits: (.commits | length)}'
   ```

## Step 3: Present summary and confirm

Show the reviewer a pre-merge summary:

```
## Ready to Merge

PR #<n>: <title>
Author: <author>
Commits: <count> (will be squash-merged into 1)
CI: All checks passing
Conflicts: None
Existing reviews: <list>

Proceed with approve + squash-merge + delete branch?
```

**Wait for explicit confirmation before proceeding.** Do not auto-approve or auto-merge.

## Step 4: Approve

```
gh pr review <PR> --approve --body "Looks good — approving and merging."
```

## Step 5: Squash-merge and delete branch

```
gh pr merge <PR> --squash --delete-branch
```

## Step 6: Confirm completion

Report the result: merged commit, branch deleted, PR closed. Link to the merged PR.
