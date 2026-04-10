---
name: verify-review
description: Verify that a PR author addressed prior review feedback — parses checklist items from the reviewer's comment and checks each against new commits.
disable-model-invocation: true
---

Verify that the author's new commits address the reviewer's prior feedback on a PR. This is the lightweight "re-review" workflow — use it when the author says "Ready for review" after pushing fixes.

## Step 1: Resolve the PR

The PR number is: $ARGUMENTS

If no PR number was provided, detect it from the current branch:
```
gh pr view --json number --jq .number
```

## Step 2: Fetch prior review comment

Identify the reviewer and find their structured feedback comment.

1. Get the current user (the reviewer running this skill):
   ```
   gh api user --jq .login
   ```

2. Fetch PR comments:
   ```
   gh pr view <PR> --json comments
   ```

3. Find the reviewer's most recent structured comment — scan comments from the current user (newest first) and select the first one that contains `### Required Changes` or checkbox items (`- [ ]`).

4. Record the comment's **body** and **createdAt** timestamp.

5. **If no structured comment found**: present the most recent comment from the reviewer and ask them to confirm which items to verify. If the reviewer has no comments on the PR at all, report this and suggest running `/review-workflow <PR>` for a first-pass review instead.

## Step 3: Identify response commits

Find commits the author pushed after the review comment.

1. Fetch the commit list:
   ```
   gh api repos/bkelly-lab/jkp-data/pulls/<PR>/commits
   ```

2. Find all commits with `commit.committer.date` after the review comment's `createdAt` timestamp. These are the "response commits."

3. Identify the diff base and head:
   - **Head**: the latest commit on the PR.
   - **Base**: start from the last commit *before* the review comment. If there is a merge commit among the response commits (the author merged main into their branch), use that merge commit as the base instead — this ensures the diff only shows the author's new work, not changes from main.

4. **If no response commits exist**: report "No new commits found after the review comment. The author may not have pushed changes yet." and stop.

## Step 4: Diff the response

Get the scoped diff covering only the response commits. Use a git worktree so the reviewer's working tree and current branch are untouched regardless of whether they have uncommitted changes.

1. Create a worktree for the PR branch. This works for fork PRs via the `pull/<PR>/head` virtual ref:
   ```
   git fetch origin pull/<PR>/head:pr-<PR>-review
   ```
   ```
   git worktree add ../jkp-data-review-pr<PR> pr-<PR>-review
   ```

2. Get the overview (run git against the worktree via `-C`):
   ```
   git -C ../jkp-data-review-pr<PR> diff <base>..<head> --stat
   ```
   ```
   git -C ../jkp-data-review-pr<PR> diff <base>..<head> --name-only
   ```

3. Get the full diff per file:
   ```
   git -C ../jkp-data-review-pr<PR> diff <base>..<head> -- <file>
   ```
   Read each changed file's diff. For large diffs, focus on the files referenced in the review items first.

## Step 5: Parse review items

Parse the review comment body into individual checklist items using a tiered approach.

**Tier 1 — Standard `/draft-pr-comment` format:**

Split the comment by `###` headings. Look for sections named "Required Changes", "Suggestions", and "Copilot Items to Address". Within each section, extract items matching:
```
- [ ] **[<location>]** <description>
```
or
```
- [ ] **[<location>] <description>**
```
Capture the category (Required/Suggestion/Copilot), location reference, and description for each item.

**Tier 2 — Generic checkboxes:**

If no standard sections are found, extract all `- [ ]` lines as items. Classify them all as "Review Items" (unknown category). Try to extract file references from bold text, backticks, or inline code.

**Tier 3 — Unstructured:**

If no checkboxes are found, present the full comment body and ask the reviewer: "This comment doesn't have a standard checklist format. Which specific items should I verify?" Wait for the reviewer to list items before proceeding.

## Step 6: Verify each item against the diff

For each parsed item:

1. Check if the referenced file appears in the response diff's changed file list.
2. If the file is in the diff, examine the relevant hunks to determine whether the change addresses the item's description.
3. Assign a verdict:
   - **Addressed**: The diff clearly addresses this item.
   - **Partially addressed**: The diff touches the relevant area but may not fully address the concern.
   - **Not addressed**: The file/area is not modified, or the changes don't relate to this item.

Present the verification as a structured summary:

```markdown
## Re-review: PR #<n>

### Review Comment
From @<reviewer> on <date> — [link to comment]

### Response Commits
<count> commit(s) after review (<base-sha>..<head-sha>), +<added>/-<removed> lines

### Checklist Verification

#### Required Changes
1. <verdict> **[file:line] Description** — <brief explanation of what changed or didn't>
2. ...

#### Suggestions
1. <verdict> **[file:line] Description** — <brief explanation>
2. ...

### Verdict
X/Y required items addressed, Z/W suggestions adopted.
```

Use checkmark/cross symbols for the verdicts to make the summary scannable.

## Step 7: New issues check and decision point

### New issues check

If the response diff includes changed `.py` files, run @code-critic on the response diff only (not the full PR diff). The changed files live inside the worktree at `../jkp-data-review-pr<PR>` — pass those paths to @code-critic. Present any findings under a "New Issues in Response" section. If no Python files changed, skip this step.

### Clean up the worktree

Before the decision point, remove the worktree and temporary ref so the reviewer's repo is left tidy:
```
git worktree remove ../jkp-data-review-pr<PR>
```
```
git branch -D pr-<PR>-review
```

### Decision point

Ask the reviewer:

> Based on the verification above, how would you like to proceed?
> (a) **Approve and merge** — run `/approve-pr <PR>`
> (b) **Request further changes** — draft a follow-up comment via `/draft-pr-comment <PR>`
> (c) **Manual review** — look at specific items more closely before deciding

If all required items are addressed and code-critic found no critical issues, recommend option (a). Otherwise, recommend option (b) and summarize what remains.

Wait for the reviewer's choice, then follow the corresponding skill.
