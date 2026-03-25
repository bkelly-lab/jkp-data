---
name: check-docs
description: Check whether code changes need documentation updates.
disable-model-invocation: true
---

Check whether code changes need documentation updates.

Gather context, then delegate to the doc-sync agent:

1. Run `git diff --name-only HEAD` to find changed files
2. Run `git diff HEAD` focusing on characteristic-related changes (`.alias()` calls, list definitions, new column names)
3. Pass the changed file list and relevant diff hunks to @doc-sync for analysis

If there are no changed Python files, report that there is nothing to check.
