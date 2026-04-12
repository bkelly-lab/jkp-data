---
name: draft-pr-comment
description: Draft and post a structured PR comment requesting specific changes from the author.
disable-model-invocation: true
---

Draft a structured PR comment requesting changes, then post it after reviewer confirmation.

## Step 1: Resolve the PR

The PR number is: $ARGUMENTS

If no PR number was provided, detect it from the current branch:
```
gh pr view --json number --jq .number
```

## Step 2: Gather findings

Use the review findings already present in this conversation (from `/triage-copilot`, `/review-pr`, or the reviewer's own observations). If no findings are available, ask the reviewer what changes they want to request.

## Step 3: Draft the comment

Compose a PR comment with this structure:

```markdown
## Review Feedback

Hi @<author>,

Thanks for this PR. A few items to address before we can merge:

### Required Changes
- [ ] **[file.py:42]** Description of what needs to change and why
- [ ] **[file.py:78]** Description of what needs to change and why

### Suggestions (optional but recommended)
- [ ] **[file.py:15]** Description of suggested improvement

### Copilot Items to Address
- [ ] **[file.py:30]** Copilot suggestion that should be incorporated (with guidance)

Let me know if you have questions about any of these items.
```

Rules for the comment:
- Be specific: name files, line numbers, and exactly what to change
- Be constructive: explain *why* each change is needed
- Use checklist format so the author can track progress
- Separate required changes from optional suggestions using these guidelines:
  - **Required**: correctness bugs, methodology issues, missing input validation, requirements mismatches with the linked issue, missing documentation for user-facing changes
  - **Suggested**: style/convention improvements, minor type annotation gaps, performance optimizations that don't affect correctness
- Include relevant Copilot suggestions that were classified as "Incorporate"
- Keep tone professional and collaborative

## Step 4: Present draft for confirmation

Show the full draft to the reviewer. Ask them to confirm or edit before posting. Do NOT post without explicit confirmation.

## Step 5: Post the comment

Once confirmed, post using:
```
gh pr comment <PR> --body "<comment>"
```

Report that the comment was posted and link to the PR.
