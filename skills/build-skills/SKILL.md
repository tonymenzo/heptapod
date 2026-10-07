---
name: build-skills
bundle: agent_dev
description: Create or improve HEPTAPOD agent skills. Use when adding guidance under `skills/`, revising a `SKILL.md`, or turning repeated agent behavior into a reusable workflow. Do not use for executable physics operations; implement those under `tools/`.
---

# Build HEPTAPOD skills

1. Define two trigger examples and one non-goal.
2. Use skills for guidance and domain knowledge; put executable operations in
   `tools/`.
3. Read the closest skill and relevant `toolkit.yaml` bundle; avoid duplication.
4. Create or update `skills/<lowercase-hyphen-name>/SKILL.md` with only `name`,
   an existing `bundle`, and a description stating what it does and when to use
   it. Keep the body imperative and concise.
5. Add `scripts/` only for repeated deterministic work, `references/` only for
   necessary detail, and `assets/` only for output material. Link each resource.
6. Run new scripts on representative inputs, then run `tb validate`.
7. Report changed files and validation; commit or push only when requested.
