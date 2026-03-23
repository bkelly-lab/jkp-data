---
name: review-code
description: Review changed Python files against project conventions.
disable-model-invocation: true
---

Review changed Python files against project conventions.

Gather context, then delegate to the code-critic agent:

1. Run `git diff --name-only HEAD` to find changed files (staged and unstaged)
2. Run `git diff HEAD` to get the full diff
3. Filter to `.py` files only
4. Pass the file list and diff summary to @code-critic for review

If there are no changed Python files, report that there is nothing to review.
