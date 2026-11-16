# D1 expansion (style-controlled AoT triplets)

Goal: bring the style-controlled subset from 10 → 20 paired triplets.

## Rules
- Third-person neutral academic prose for all three framings.
- Approximately equal length across the three framings (within ~15%).
- No first-person narration, no emotional vocabulary, no rhetorical questions.
- The same AoT moral event must be described in all three framings.
- Only the *moral perspective foregrounded* differs:
  - sympathetic → actor's intentions, constraints, justifications
  - condemning → victims, harms, consequences
  - abstract → entities replaced with role nouns; no narrative voice; no group identities

## Required fields (JSONL row)
```json
{
  "id": "sc-011",
  "dilemma": "<short label>",
  "source": "<arc / chapter>",
  "decision_maker": "<name>",
  "ethical_frameworks": ["<f1>", "<f2>"],
  "sympathetic": "<3rd-person neutral, foregrounding actor reasoning>",
  "condemning":  "<3rd-person neutral, foregrounding harms / victims>",
  "abstract":    "<role nouns only, no narrative voice>"
}
```

## ID convention
`sc-001` … `sc-010` are existing. New entries continue at `sc-011` … `sc-020`.

## Curation workflow
1. Pick a triplet from D0 that the existing controlled set does not yet cover.
2. Draft three framings adhering to the rules above.
3. Length-match (target 700–900 characters per framing).
4. Append to `aot_style_controlled_v2.jsonl`.
5. Run `pytest tests/test_data_shapes.py -k style_controlled` to confirm schema.
