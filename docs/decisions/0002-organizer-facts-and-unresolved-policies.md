# ADR 0002 — Organizer-confirmed facts and intentionally unresolved policies

- **Status:** Open (living registry; resolved items graduate to confirmed facts)
- **Date:** 2026-07-15 (Audit 0008, Phase 1C-A)
- **Supersedes:** extends ADR 0001 (does not replace it)
- **Context:** Phase 1C-A must build the foundation for downloading and using
  external datasets and KBs (RxNorm, Vietnamese ICD-10, public NER corpora)
  *before* any data exists. The organizer discloses some conventions and
  deliberately hides others. To avoid hard-coding guesses, every fact and every
  open policy is recorded as data in `configs/organizer/*.yaml` and loaded by
  `mednorm_vi.organizer_policy`.

## Decision

1. **Separate confirmed facts from hypotheses, permanently.**
   `confirmed_facts_v1.yaml` holds ONLY organizer-confirmed statements (each with
   its confirming source). Everything undisclosed lives in
   `unresolved_policies_v1.yaml` or a hypotheses file. Code and docs must never
   present a hypothesis as a confirmed rule. The loader enforces the split
   (`PolicyHypothesis.is_confirmed` is always `False`).

2. **A policy resolves only by evidence.** Each `PolicyHypothesis` records
   status, supporting observation, contradicting evidence, confidence, a test
   method, and a `leaderboard_experiment_id` (filled once a probe exists). Until
   then MedNorm-VI uses a documented `internal_default` that carries no claim
   about the organizer's actual choice.

3. **Confirmed facts encoded (non-exhaustive).** External datasets usable for
   training subject to license; the team may need to provide data/generation
   process; RxNorm is a 2026 version; ICD-10 is Vietnamese; the organizer will
   not provide the KB files; TEST_NAME and TEST_RESULT may occur independently;
   TEST_RESULT need not be numeric; pairing is not required in output; repeated
   symptoms are distinct concepts; assertions are multi-label; the published
   offset example is end-exclusive over code points; the submission layout and
   labels are fixed.

4. **Key unresolved policies encoded.** Position coordinate space, char-vs-byte,
   line-ending behavior, interval convention, matching mode; exact RxNorm release,
   full-vs-prescribable distribution, active/suppressed/historical handling,
   legacy remapping, TTY priority, missing-strength fallback, range-strength,
   combination-drug; ICD source/version, dotted-vs-undotted, specificity;
   descriptive TEST_RESULT boundary; procedure handling; symptom boundary;
   isHistorical semantics.

## Consequences

- No module may treat any RxNorm/ICD snapshot as the organizer's exact KB, or any
  position policy as the organizer's global coordinate space.
- The internal invariant `original_text[start:end] == text` stays strict; a
  submission position is a *separate, reversible* encoding (see
  `docs/position/POSITION_POLICY.md`).
- Resolving a policy is a future leaderboard experiment, recorded back into the
  registry, then (only if organizer-confirmed) graduated to a confirmed fact.

## Convention

Add a numbered ADR for any architecture-sensitive choice; never delete one.
When a policy resolves, update its entry (status + experiment id) and, if the
organizer confirms it, move the statement into `confirmed_facts_v1.yaml`.
