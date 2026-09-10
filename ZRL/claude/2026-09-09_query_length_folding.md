# Query-length folding for Spyre paged attention

Date: 2026-09-09

This note isolates the query-axis investigation from the independent KV block
scaling work.

## Goal

Use `(KV head, query row)` as the logical attention ownership domain without
physically replicating K/V over query rows.

For Granite 3.3:

```text
Q=1:   8 * 1   = 8 independent rows
Q=512: 8 * 512 = 4096 independent rows, capped at 32 cores
```

## Results

Head-major layout with eight KV pages on `SPYRE_DEVICES=1`:

| Shape | 8 cores | 32 cores | HBM pool 8 -> 32 |
|---|---:|---:|---:|
| Decode `Q=1` | **781.3 us** | 1771.1 us | 66 -> 578 KB |
| Prefill `Q=512` | 7230.2 us | **6471.2 us** | 16512 -> 17664 KB |

All runs were numerically correct.

The production PR #4347 microbenchmark at `Q/KV=1/1024` and `512/1024`
confirmed the same policy on `SPYRE_DEVICES=1`:

| Shape | Automatic PR #4347 | Forced 8-core cap | Query ownership benefit |
|---|---:|---:|---:|
| Decode `Q=1`, KV=1024 | 1056.4 us | **1054.1 us** | none (-0.2%) |
| Prefill `Q=512`, KV=1024 | **5018.4 us** | 5038.9 us | 0.4% |

The production difference is small at eight 128-token pages because both plans
already have enough work and the 8-core plan processes larger contiguous query
regions. The standalone repro with 64-token pages showed a larger 10.5% benefit,
so the gain depends on page geometry and graph structure rather than query length
alone.

For `Q=512`, PR #4347 already selects the query dimension as the primary output
split. The score BMM, softmax chain and value BMM all run on 32 cores. No new KV
cache layout is required.

For `Q=1`, the query axis contributes no work. Forcing 32 cores splits less
appropriate axes, increases HBM traffic and more than doubles latency. Batch-one
decode should remain at the eight-KV-head ownership domain.

## Explicit flattening result

The standalone repro can set:

```bash
QUERY_FOLD=1
```

It explicitly flattens `(KV head, query row)` into the BMM batch and expands K/V
over query rows. At `Q=512` it remained correct but increased the HBM pool to
136448 KB, versus 17664 KB for the ordinary BMM. The expanded K/V ownership was
not representable by downstream consumers.

Measured latency for the identical eight-page standalone graph was:

```text
ordinary logical query ownership:   6.496 ms
explicit K/V-expanded query folding: 207.435 ms
```

Explicit flattening is therefore 31.9x slower and is retained only as a negative
reproduction. It must not be enabled in production.

Therefore query folding must remain a logical work-division choice. K/V should
be broadcast implicitly across query rows rather than physically cloned.

## Recommended policy

```text
Q=1:
    head-major KV ownership
    split over 8 KV heads

Q=512:
    head-major storage
    split over query rows and KV heads
    use all 32 cores
```

The next useful query-axis change is not more parallelism. It is a counted query
loop over smaller tiles, for example four 128-row tiles, with max/sum/output kept
in LX while each tile scans all KV pages. This preserves 32-core query ownership
while reducing the live query and output state.

## Folding query partitions into page fetching

The next experiment replicated each physical `(page, KV head)` index once per
query partition and made the page gather itself produce the BMM batch:

```text
Q=512, P partitions
query tile length = 512 / P
gather rows = P * 8 KV heads
```

K used the already-transposed cache and V used head-major storage, avoiding the
K restickify that rejected local reads in the ordinary head-major path. Results:

| Query partitions | Gather cores | Query rows/partition | Latency | HBM pool |
|---:|---:|---:|---:|---:|
| 1 | 8 | 512 | **6042.9 us** | **13568 KB** |
| 2 | 16 | 256 | 6776.9 us | 13824 KB |
| 4 | 32 | 128 | 7514.1 us | 14336 KB |
| 8 | 32 | 64 | 8877.1 us | 18432 KB |

All variants were numerically correct. Folding query partitions into page fetch
is therefore non-beneficial on the current stack:

```text
P=2: 12.1% slower
P=4: 24.3% slower
P=8: 46.9% slower
```

The four-way path does achieve its intended cardinality: K and V gathers run on
32 cores and the score BMM runs on 32 cores. It also reduces the pool versus the
ordinary head-major baseline, but not versus the matching transposed-K baseline.
The replicated gather performs four real page reads and increases index/layout
work. More importantly, its physical owner order still does not match the BMM:

```text
fetch: 32 rows owned by the gathered row/device axis
BMM:  query-row-major and KV/head/output-stick compound ownership
```

The planner reports unrepresentable or broadcast-read mismatches at those
boundaries, so the replicated data is not consumed as one local page per core.
Eight partitions can pin the K gather itself, but still cannot align its clone
and BMM owner order and is the slowest variant.

This narrows the viable compiler solution: do not issue extra HBM page fetches
per query partition. Fetch each K/V page once, then emit an LX-local broadcast
from the eight page-head owners to the 32 query-owned BMM cores. The existing
relayout planner currently rejects this chain because the page goes through a
restickify/pointwise node and the grouped destination owner ordering is not an
even broadcast.

A useful torch-spyre enhancement would let PR #4347 jointly select:

```text
8-core page gather
8-core K restickify or transposed-K gather
certified 8 -> 32 LX broadcast
32-core (query row, KV head) score/value BMM
```

and price that against the current HBM-mediated path. The transfer must preserve
the BMM's exact query/KV owner order; matching only core counts is insufficient.

## Torch-spyre multisource LX broadcast prototype

The reason query-partitioned fetching performs multiple HBM reads is explicit in
the index table: repeating eight `(page, KV head)` indices `P` times makes
`index_select` produce `P*8` real rows. Each output row is populated from HBM;
the operation is not a metadata alias or a device broadcast. Four query
partitions therefore request four copies of every page-head payload.

The correct operation is an all-gather/broadcast:

```text
8 page-head owners collectively hold one complete K/V page
each of 32 query-parallel consumers needs that complete page
```

This is not a simple one-source broadcast. Every destination receives data from
all eight sources (`fanin=8`), and every source sends to all 32 destinations
(`fanout=32`). The existing relayout legality check required `fanin=1`, so it
correctly rejected the operation it did not know how to emit.

A gated torch-spyre prototype now enables exact uniform multisource routing:

```bash
SPYRE_LX_MULTISOURCE_BROADCAST=1
LAYOUT_SOLVER=greedy
```

The proof requires:

```text
complete unique source ownership
every source and destination participates
uniform fan-in and fan-out
uniform destination replication where required
conservation of every ownership-intersection edge
exact PerCoreView-derived source-to-destination routes
```

The routes are serialized through the existing `prodConsList` shuffle contract.
OpSpec validation was generalized from completed-reduction one-source routes to
uniform explicit routes while remaining fail-closed for missing or nonuniform
destinations.

### Measurements

Matching transposed-K/head-major-V Q=512 graph, eight pages:

| Variant | Latency | HBM pool |
|---|---:|---:|
| Default CP-SAT, no relayout | 6042.9 us | 13568 KB |
| Greedy, no multisource broadcast | **4897.4 us** | 5376 KB |
| Greedy, multisource LX broadcast | 5623.0 us | **5120 KB** |

The multisource path is correct and eliminates the page-fetch ownership failures:
K/V gathers and score/value BMMs are all LX-resident. The HBM pool falls by 256
KB versus the matching greedy baseline and by 8.4 MB versus the default solver.

However, the explicit all-gather shuffles add 14.8% latency compared with the
matching greedy baseline. Eliminating HBM does not yet compensate for issuing an
LX shuffle for every K and V page/group boundary. A named batch/query BMM-owner
attempt gave 5690.6 us and did not improve the route.

For decode `Q=1`, enabling the gate remains correct and slightly improves the
same transposed-K graph:

```text
previous 8-core measurement: 781.3 us, 66 KB
multisource-gated greedy:     763.9 us, 64 KB
```

This does not prove the multisource operation itself helps decode; most accepted
decode relayouts are ordinary one-source broadcasts already legal before the new
gate.

### Interpretation

The desired query parallelism is valid and torch-spyre can now express the
necessary communication without rereading HBM. The remaining problem is cost,
not correctness:

1. The full page is all-gathered separately for multiple unrolled query groups.
2. K and V each require their own shuffle.
3. The page loop is unrolled, so transfer setup repeats per page.
4. PR #4347's current cost model underprices route fan-in/fan-out and does not
   yet decide that the HBM baseline is faster for this shape.

Further improvement should share one all-gathered K/V destination across all GQA
groups and, ideally, retain it while all query groups consume that page. A
page-counted loop could then perform one K and one V all-gather per page rather
than materializing repeated shuffle operations in the unrolled graph. The
placement-aware selector must price route edges and shuffle synchronization so
the feature is selected only when measured/predicted beneficial.

### Follow-up implementation

The existing planner already groups all equivalent GQA consumers of one page
source into one destination allocation. There are therefore two shuffles per
page (one K and one V), not one shuffle per query group. Across eight pages the
prototype emits sixteen multisource shuffles. Sharing across GQA was already
realized by the atomic consumer grouping in `collect_lx_relayout_plans()`.

A packed K/V page prototype attempted to reduce this to one shuffle per page by
gathering `[KV, 2, block, D]`. It was correct but the K and V selections became
partial/offset reads of the packed buffer, and K additionally crossed a
restickify. Exact whole-buffer relayout certification therefore did not apply:

```text
latency: not pursued after placement failure
HBM pool: 9472 KB
buf0: partial/offset read or restickify local-read failure
```

Supporting packed transfer would require sub-buffer LX ownership and a shuffle
that can route selected physical regions under two different downstream layouts.
That is substantially larger than one shared-copy optimization and is not the
next minimal change.

The placement-aware cost model was extended with `relayout_fanin`. Ordinary
shuffle/broadcast has fan-in 1; this all-gather has fan-in 8 and is charged for
eight times the per-destination payload. The allocator now trials the graph with
multisource plans enabled and disabled and regenerates the cheaper plan cleanly.

For this Q=512 shape the selector predicts:

```text
multisource enabled:  12.373 ms
multisource disabled: 12.250 ms
```

and correctly chooses the no-all-gather graph. The selected graph measured
5.168 ms and retained the 5376 KB HBM pool. This is slightly above the earlier
4.897 ms sample but avoids the 5.623 ms multisource regression by construction.
The exact medians vary between compile/run samples; the selected ordering is
consistent with the measured direction.

The remaining performance route is a page-counted loop that amortizes one K/V
communication setup over repeated page consumption, or backend support for a
fused gather-and-broadcast operation. Until then, multisource LX broadcast stays
experimental and cost-selected rather than unconditional.

## Reproduction files

```text
.claude-scratch/repro_lx_residency.py
scripts/microbench/configs/granite33_8b_bs128_query_fold.json
```

The production microbenchmark config contains only the two required cases,
`Q=1` and `Q=512`, both with 1024 KV tokens.
