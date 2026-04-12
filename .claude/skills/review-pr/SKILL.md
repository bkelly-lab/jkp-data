---
name: review-pr
description: Run all automated PR checks — code conventions, documentation sync, and holistic review.
disable-model-invocation: true
---

Run a comprehensive automated review of a pull request using three specialized agents.

## Step 1: Resolve the PR

The PR number is: $ARGUMENTS

If no PR number was provided, detect it from the current branch:
```
gh pr view --json number --jq .number
```

## Step 2: Gather PR context

Run these commands:

1. **PR metadata**:
   ```
   gh pr view <PR> --json title,body,author,headRefName,baseRefName,mergeable,mergeStateStatus
   ```

2. **CI status**:
   ```
   gh pr checks <PR>
   ```

3. **Full diff**:
   ```
   gh pr diff <PR>
   ```

4. **Changed file list**:
   ```
   gh pr diff <PR> --name-only
   ```

5. **Linked issue** (if referenced in PR body — look for "Fixes #N", "Closes #N", "Solution to #N", "#N"):
   ```
   gh issue view <N> --json title,body
   ```
   Pass the issue context to @pr-reviewer so it can verify the implementation matches the original requirements.

If the PR has merge conflicts (`mergeable` is `CONFLICTING`), report this immediately and stop — the author needs to resolve conflicts before review can proceed.

Filter the changed files to `.py` files for code-critic and doc-sync. Pass all files to pr-reviewer.

## Step 3: Run three agents in parallel

Delegate to all three agents, passing the PR diff and metadata to each:

1. **@code-critic** — Review changed Python files against CLAUDE.md conventions (C1–C10 + ruff). Scope the review to the PR diff (not local working tree). Use `gh pr diff <PR>` as the diff source, not `git diff HEAD`.

2. **@doc-sync** — Check whether code changes need documentation updates. Pass characteristic-related changes from the PR diff.

3. **@pr-reviewer** — Holistic review: correctness, methodology, architecture, test coverage, performance, completeness, data impact.

## Step 4: Synthesize results

Combine the three agent reports into a unified review:

```
## PR #<n> Review Summary

### Convention Checks (code-critic)
[summary of critical/important/advisory findings]

### Documentation Sync (doc-sync)
[summary of documentation coverage]

### Holistic Review (pr-reviewer)
[summary of correctness, methodology, architecture, etc.]

### CI Status
[pass/fail/pending]

### Overall Verdict
[APPROVE / REQUEST_CHANGES / NEEDS_DISCUSSION]
[Key action items, if any]
```
