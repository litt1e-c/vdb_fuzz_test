# Qdrant Logic Report

- Report file: `QDRANT_LOGIC_REPORT__server-v1.17.0__client-v1.16.1__transport-rest+grpc__date-2026-04-01.md`
- Report date: `2026-04-01`
- Scope: Qdrant filter logic semantics, current local scripts, and `qdrant_fuzz_oracle.py`

## 1. Naming Convention

### 1.1 Markdown report naming

Recommended pattern:

`<ENGINE>_LOGIC_REPORT__server-v<SERVER_VERSION>__client-v<CLIENT_VERSION>__transport-<TRANSPORT>__date-<YYYY-MM-DD>.md`

Example:

`QDRANT_LOGIC_REPORT__server-v1.17.0__client-v1.16.1__transport-rest+grpc__date-2026-04-01.md`

### 1.2 Test / probe case naming

Recommended pattern:

`<CATEGORY>__<SHAPE>__<PREDICATE>`

Examples:

- `ATOM__must__score_eq_10`
- `NULL__must__score_is_null`
- `COUNT__must__misc_values_lt_1`
- `LOGIC__must__score_gt_0_or_not_score_gt_0`

### 1.3 Bug POC naming

Recommended pattern:

`poc_qdrant_<issue-or-unfiled>_<short_slug>.py`

Examples:

- `poc_qdrant_issue8538_int64_float_precision_loss.py`
- `poc_qdrant_unfiled_int64_closed_range_aliasing.py`

## 2. Environment

- Server: `Qdrant 1.17.0`
- Server commit: `4ab6d2ee0f6c718667e553b1055f3e944fef025f`
- Client: `qdrant-client 1.16.1`
- Python: local runtime used by current workspace
- Transport checked: `REST` and `gRPC`
- Local deployment source: `deploy_qdrant_v1_17_0/docker-compose.yml`

Primary Qdrant docs used for interpretation:

- Filtering concepts: <https://qdrant.tech/documentation/concepts/filtering/>

Important official statements from that page:

- Qdrant exposes boolean clauses `AND`, `OR`, `NOT` via `must`, `should`, `must_not`.
- `IsEmpty` matches `missing`, `null`, or `[]`.
- `IsNull` matches only fields that exist and are `NULL`.
- `values_count` treats non-array stored values as count `1`.

## 3. Bottom Line

### Final classification

For Qdrant `v1.17.0`, the **core boolean composition is 2-valued set logic**, not SQL-style 3VL.

But the full system is not "plain 2VL everywhere" either:

- Boolean composition of ordinary predicates behaves like 2VL.
- Qdrant also provides **special null-aware predicates** such as `is_null` and `is_empty`.
- `values_count` has its own semantics:
  - explicit `null` behaves like count `0`
  - `[]` behaves like count `0`
  - scalar non-array behaves like count `1`
  - truly missing field does **not** participate

So the most accurate statement is:

> Qdrant is **not SQL 3VL**. Its filter engine behaves as **2VL set logic plus predicate-specific null handling**.

If you must choose between "2VL / 3VL / both", my recommendation is:

- Core logic answer: `2VL`
- System-level answer: `2VL core + null-aware special predicates`, not true `3VL`

## 4. Evidence

I rewrote `test_2vl_qdrant.py` into a reusable probe that separates:

- missing field
- explicit null
- empty array
- scalar value
- nested object array
- boolean tautologies that distinguish 2VL from 3VL

Key observed results on both `REST` and `gRPC`:

| Probe | Actual result | Meaning |
| --- | --- | --- |
| `ATOM__must_not__score_gt_0` | keeps missing/null rows | `NOT(A)` keeps rows where `A` does not match |
| `LOGIC__must__score_gt_0_or_not_score_gt_0` | returns all rows, including missing/null | `A OR NOT A` is true for missing/null under Qdrant filter semantics |
| `NULL__must__score_is_null` | matches only explicit null | `is_null` is not "missing or null" |
| `NULL__must__score_is_empty` | matches missing + explicit null | `is_empty` collapses missing and null |
| `COUNT__must__misc_values_lt_1` | matches explicit null and `[]`, not missing | `values_count` is predicate-specific, not generic 2VL/3VL |
| `NESTED__must_not__items_a_eq_1` | keeps missing/null/[] nested rows | nested outer `must_not` follows complement behavior |

The strongest discriminator is:

- `A OR NOT A` returned **all rows**, including missing/null rows.

Under SQL 3VL, rows where `A` is `UNKNOWN` would not satisfy `A OR NOT A`.
So this result rules out true SQL-like 3VL for Qdrant's core boolean composition.

## 5. Why This Is Not 3VL

If Qdrant were implementing SQL 3VL for ordinary comparison predicates:

- `score > 0` on `null` would be `UNKNOWN`
- `NOT(score > 0)` on `null` would still be `UNKNOWN`
- `score > 0 OR NOT(score > 0)` on `null` would still be `UNKNOWN`

But Qdrant actually returns the missing/null rows for `NOT(score > 0)` and for `score > 0 OR NOT(score > 0)`.

That is 2VL-style complement behavior over the match set.

## 6. Current `test_2vl_qdrant.py` Review

### 6.1 Original script problems

The original script was useful as an initial sketch, but it was not yet a reliable decision tool.

Main problems:

- It left several expected results as `?`, so many rows were never really asserted.
- It had at least one wrong expectation:
  - `MatchAny tags contains 1` expected `[1, 5]`
  - actual correct result is not `[1, 5]`
- It did not distinguish clearly between:
  - missing field
  - explicit null
  - empty array
- It did not test `is_null` vs `is_empty`.
- It did not include a real 2VL-vs-3VL discriminator such as `A OR NOT A`.
- It printed warnings but did not fail the process with a non-zero exit code.
- It did not print server/client version information.

### 6.2 Current upgraded script

The upgraded `test_2vl_qdrant.py` now:

- prints server version and commit
- prints client version
- covers `missing / explicit null / [] / scalar / nested`
- tests `is_null`, `is_empty`, `values_count`
- tests `A OR NOT A`, `A AND NOT A`, and double negation
- returns non-zero on mismatch
- validates the same conclusions on both `REST` and `gRPC`

## 7. Fuzzer Assessment

Target file:

- `qdrant_fuzz_oracle.py`

### 7.1 What is implemented well

The fuzzer is generally careful and much better than a naive random generator.

Good parts:

- It explicitly models `must_not` as complement-style behavior for null/non-match paths.
- It treats `MatchExcept` as not matching null/missing, which matches observed Qdrant behavior.
- It distinguishes `is_null` and `is_empty`.
- For schema evolution fields, it intentionally uses `IsEmpty` semantics by default, which is a good practical workaround because evolved fields are often truly missing rather than explicitly null.
- It already warns about known `float32` boundary problems and recommends `--prefer-grpc`.
- It has equivalence mode, which is valuable for checking logical rewrites.

Runtime sanity checks:

- `python qdrant_fuzz_oracle.py --oracle --rounds 20 --seed 42 --prefer-grpc`
  - passed
- `python qdrant_fuzz_oracle.py --equiv --rounds 12 --seed 43 --prefer-grpc`
  - passed
- `python qdrant_fuzz_oracle.py --oracle --dynamic --rounds 12 --seed 44 --prefer-grpc`
  - passed

### 7.2 Where the fuzzer is still incomplete

For "logic classification" specifically, the main gap is:

- Base data mostly writes every schema field with explicit `None`, rather than omitting fields entirely.

That means:

- explicit null semantics are well covered
- truly missing-field semantics are **not** broadly covered in normal base-schema tests
- missing-field semantics are mostly exercised through:
  - schema evolution `evo_*` fields
  - manually crafted probes
  - some nested/JSON paths

So if the goal is specifically "Qdrant sees missing as what?", the fuzz oracle alone is **not sufficient**. A dedicated probe like the upgraded `test_2vl_qdrant.py` is still needed.

### 7.3 Important practical limitation

Short dynamic fuzz runs may not hit schema evolution at all, because schema evolution is only triggered every 30 rounds in oracle mode.

That means:

- `--dynamic --rounds 10` or `12` is not enough to judge evolved-field missing semantics
- you need `>= 30` rounds to cover that path

### 7.4 Conclusion on the fuzzer

My judgment:

- As a general correctness fuzzer for Qdrant filters, it is **overall well implemented**.
- As a tool for **proving 2VL vs 3VL by itself**, it is **not fully sufficient**, because missing-field coverage is not wide enough in the base dataset.
- Best practice is:
  - use the dedicated logic probe for semantics classification
  - use the fuzz oracle for broader regression and differential checking

## 8. Why One Dynamic Fuzz Run Failed

Command:

- `python qdrant_fuzz_oracle.py --oracle --dynamic --rounds 32 --seed 45 --prefer-grpc`

Result:

- one mismatch at case 27

Expression:

`NOT ((NOT (NOT (c1 == False)) AND (c2 >= 9223372036854775806 AND NOT (c6 == False))))`

At first glance this looks like a logic failure, but after reduction and minimal repro it is **not primarily a 2VL/3VL issue**.

Root cause:

- the atomic predicate `c2 >= 9223372036854775806` itself is wrong near the int64 boundary
- once the atomic range predicate is already wrong, the outer `NOT` and `AND` composition will also look wrong

So this mismatch should be classified as a **boundary comparison bug**, not as evidence of 3VL.

## 9. Known Bugs That Must Be Separated From Logic Analysis

### 9.1 Existing bug: int64 open interval

POC:

- `history_find_bug/qdrant/poc_qdrant_issue8538_int64_float_precision_loss.py`

Observed:

- value inserted: `-9223372036854775807`
- exact closed range returns `[1]`
- open interval `(VALUE-1, VALUE+1)` incorrectly returns `[]`

Meaning:

- near-int64-boundary range evaluation is unreliable

### 9.2 Existing bug: exact `float32_max` closed range

POC:

- `qdrant_bug_float32_boundary.py`

Observed:

- exact closed range at `+/-FLOAT32_MAX` returns empty
- `nextafter` bracketed range returns the correct point

Meaning:

- exact scalar range checks at `+/-float32_max` are unreliable

### 9.3 Newly isolated bug: closed-range aliasing near int64 extremes

New POC added:

- `history_find_bug/qdrant/poc_qdrant_unfiled_int64_closed_range_aliasing.py`

Observed on both `gRPC` and `REST`:

- `x >= -9223372036854775807` incorrectly includes `-9223372036854775808`
- `x == -9223372036854775807` incorrectly matches three values
- `x >= 9223372036854775806` incorrectly includes `9223372036854775805`
- `x == 9223372036854775806` incorrectly matches two values
- `x > 9223372036854775805` incorrectly matches nothing
- `x <= 9223372036854775805` incorrectly includes `9223372036854775806`

Meaning:

- near-int64-extreme range comparisons are not just imprecise; they can alias adjacent values or collapse unexpectedly
- this bug can easily masquerade as a boolean logic failure in fuzz output

## 10. Final Verdict

### About Qdrant logic

Qdrant `v1.17.0` should be described as:

- `2VL` for core boolean filter composition
- plus explicit null-aware predicates such as `is_null` and `is_empty`
- plus special predicate behavior such as `values_count`
- not SQL-style `3VL`

### About the current test coverage

The original `test_2vl_qdrant.py` was **not complete enough** to settle the question rigorously.
The upgraded version is much closer to a reliable decision probe.

### About the fuzzer

`qdrant_fuzz_oracle.py` is **overall good**, but:

- it is better at broad regression than at pure semantics classification
- it needs dedicated probes to cover true missing-field behavior
- boundary-value mismatches must be triaged separately from logic mismatches

### Recommended workflow going forward

1. Use `test_2vl_qdrant.py` to classify semantics.
2. Use `qdrant_fuzz_oracle.py --prefer-grpc` for broader fuzz regression.
3. Treat any case involving near-int64 extremes or `+/-float32_max` as "boundary bug candidate first, logic bug second".
4. If you want a "logic-only" fuzz campaign later, add a mode that excludes extreme int64/float boundary injections.
