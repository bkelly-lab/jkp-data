---
name: review-workflow
description: Full PR review workflow — Copilot triage, automated checks, change requests or approval.
disable-model-invocation: true
---

Run the full PR review workflow end-to-end. This guides you through setup, automated analysis, and a decision point (request changes or approve).

The PR number is: $ARGUMENTS

If no PR number was provided, ask the reviewer for it.

---

## Phase 1 — Setup

### 1.1 Fetch PR metadata

```
gh pr view <PR> --json number,title,body,author,headRefName,baseRefName,mergeable,mergeStateStatus,statusCheckRollup
```

Display a summary: PR number, title, author, branch, CI status.

### 1.2 Check for blockers

- If `mergeable` is `CONFLICTING`: report merge conflicts and **stop**. The author must resolve them.
- If CI checks are failing: report which checks failed. Continue with review (but note failures).

### 1.3 Ensure Copilot review exists

Check for existing Copilot review:
```
gh api repos/bkelly-lab/jkp-data/pulls/<PR>/reviews --jq '[.[] | select(.user.login == "copilot-pull-request-reviewer[bot]")] | length'
```

- If **0**: Request a review from Copilot and inform the reviewer:
  ```
  gh pr edit <PR> --add-reviewer copilot
  ```
  Tell the reviewer: "Copilot review has been requested. It may take a minute. Continuing with automated checks in the meantime — Copilot triage will run once its review arrives."

- If **>0**: Copilot review already exists. Proceed.

---

## Phase 2 — Automated Analysis

Run these two analyses. If Copilot review is already available, run both. If Copilot review was just requested, run `/review-pr` and the self-check first, then check for the Copilot review before presenting combined findings. If Copilot still hasn't responded by then, proceed without Copilot triage — note this in the combined findings, and fold Copilot's feedback in later if the review arrives before the final comment is posted.

### 2.1 Copilot Triage

Follow the `/triage-copilot` skill instructions for this PR. This fetches Copilot's inline comments and overview, then delegates to @copilot-triage for classification.

### 2.2 Automated Review

Follow the `/review-pr` skill instructions for this PR. This runs @code-critic, @doc-sync, and @pr-reviewer in parallel and synthesizes results.

### 2.3 Resolve Copilot threads

After triaging Copilot's suggestions, resolve all Copilot review threads on the PR since we've incorporated the findings into our own review. Use the GraphQL API:

1. Fetch thread IDs:
   ```
   gh api graphql -f query='{ repository(owner: "bkelly-lab", name: "jkp-data") { pullRequest(number: <PR>) { reviewThreads(first: 20) { nodes { id isResolved comments(first: 1) { nodes { author { login } } } } } } } }'
   ```

2. For each unresolved thread from `copilot-pull-request-reviewer[bot]`, resolve it:
   ```
   gh api graphql -f query='mutation { resolveReviewThread(input: {threadId: "<THREAD_ID>"}) { thread { isResolved } } }'
   ```

Run all resolve mutations in parallel.

### 2.4 Self-check: run tests and linter

Run tests and linting on the PR's changed files as a concrete verification step — don't just trust CI status.

0. Check out the PR branch so the changed files are available locally. Record the current branch so you can switch back afterward:
   ```
   gh pr checkout <PR>
   ```

1. Identify changed Python files from the diff (step 2.2 already has this).

2. Run linter on changed production code:
   ```
   uv run --group lint ruff check <changed .py files in code/>
   ```

3. If the PR includes new or changed test files, run them:
   ```
   uv run --group test pytest <changed test files> -v --no-header
   ```

4. Include pass/fail results in the combined findings. If tests fail locally, flag this even if CI says "pass" — it may indicate environment-dependent behavior.

5. Switch back to the original branch:
   ```
   git checkout <original branch>
   ```

### 2.5 Present combined findings

Show the reviewer a combined summary:

```
## PR #<n> — Combined Review Findings

### Copilot Triage
- X suggestions to incorporate, Y to ignore, Z to discuss
[key items listed]

### Convention Checks
[critical/important/advisory counts]
[key items listed]

### Documentation Sync
[status]

### Holistic Review
[key findings per dimension]

### CI Status
[pass/fail/pending]
```

---

## Phase 3 — Interview & Decision

### 3.0 Interview the reviewer

Before asking for a decision, ask 2–3 probing questions based on the review findings. The goal is to surface issues the automated review may have missed by engaging the reviewer's domain expertise. Tailor questions to the specific PR — examples:

- "The PR changes how X is computed — does this align with the methodology in the paper?"
- "This function doesn't handle edge case Y — is that intentional or an oversight?"
- "The PR description says 'no data impact' but column Z is added/removed — is that accurate?"
- "This PR depends on PR #N which is still open — should we merge that first?"
- "The winsorization bounds are hardcoded at 0.1%/99.9% — should these be configurable?"

Wait for the reviewer's answers. Incorporate any new insights into the findings before proceeding.

### 3.1 Decision point

Ask the reviewer:

> Based on the findings above, how would you like to proceed?
> (a) **Request changes** — I'll draft a PR comment for the author
> (b) **Approve and merge** — I'll run pre-flight checks and merge
> (c) **Add your own observations** — tell me what else you noticed and I'll factor it in

### If (a) — Request changes

Follow the `/draft-pr-comment` skill instructions. Draft a structured comment incorporating:
- Findings from Copilot triage (items marked "Incorporate")
- Critical and important findings from code-critic
- Key findings from the holistic review
- Any reviewer observations from option (c)

After posting the comment, tell the reviewer:
> "Comment posted on PR #<n>. Run `/review-workflow <n>` again after the author pushes changes."

### If (b) — Approve and merge

Follow the `/approve-pr` skill instructions. Run pre-flight checks, present summary, confirm, then approve and squash-merge.

### If (c) — Add observations

Ask the reviewer to describe their observations. Incorporate them into the review findings, then re-ask options (a) or (b).

---

## Phase 4 — Re-review (when re-invoked on the same PR)

When this skill is run on a PR that already has review comments from the reviewer:

1. **Detect new commits** since the last review:
   ```
   gh api repos/bkelly-lab/jkp-data/pulls/<PR>/commits --jq '.[-1].commit.committer.date'
   ```
   Compare with the timestamp of the last review comment.

2. **Skip Copilot** — do not re-request a review. Copilot triage was done in the first pass.

3. **Run `/review-pr`** focused on changes since the last review. Note which previous findings have been addressed.

4. **Return to Phase 3** — present updated findings and ask the reviewer to decide.
