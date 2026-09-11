#!/usr/bin/env python3
"""LX vs HBM residency of the per-sequence paged-attention kernel.

Standalone: torch and torch_spyre only. The kernel, the KV-cache store and the
page-cache device layout are inlined below, so nothing is imported from
spyre-inference (or vLLM) and the script can be handed to torch-spyre as-is.

Reports, for every buffer the kernel allocates:

  * LX or HBM per allocation, from the generated SDSC.
  * the layout planner's per-op verdict and reason, from
    `lx_pinning: <buf> (<kind>) -> <reason>` (scratchpad.allocator at DEBUG).
  * the per-arg layouts the cost model predicted (SPYRE_DUMP_COST).
  * the HBM spill pool the kernel asks for (bundle.mlir device_mem_allocate).

The result is checked against a CPU run of the same closure.

Buffers of interest, in the kernel's own terms:
    k_page / v_page   the gathered KV page   (index_select -> permute)
    scores            q @ k_page^T
    tile_probs        exp(scores - max)      = the "P" operand of the second matmul
    tile_output       tile_probs @ v_page

The KV write is included by default (WRITE_KV=1): `index_copy_` into slot-major
views of the pages, compiled as its own graph, so one run covers the store as
well as the read side. It appears as a second kernel in the report.

Defaults mimic a granite-3.3-8b decode step: 32 query heads over 8 KV heads,
head_size 128, block_size 64, one query row, four active pages.

Env knobs:
  NUM_BLOCKS=4  Q_LEN=1  KV_HEADS=8  QPK=4  HEAD_SIZE=128  BLOCK_SIZE=64
  NUM_PAGES=32                      cache pages; sets the gather's source size,
                                    which does not move the LX verdicts
  WRITE_KV=0                        skip the KV store, read side only
  LAYOUT_SOLVER=cpsat|greedy        passed through to torch-spyre
  OUT_DIR=<dir>                     artifacts land here (default: fresh tmp dir)
  VERBOSE=1                         per-op verdicts and unclassified buffers

Each run gets a fresh TORCHINDUCTOR_CACHE_DIR: a warm cache skips compilation and
emits neither SDSC nor planner logs.
"""

import json
import os
import re
import statistics
import tempfile
import time
from pathlib import Path

OUT = Path(os.environ.get("OUT_DIR") or tempfile.mkdtemp(prefix="lx_residency_"))
OUT.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT / "inductor-cache"
PLANNER_LOG = OUT / "planner.log"
COST_FILE = OUT / "cost.txt"

# Must all precede the torch import.
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(CACHE_DIR)
os.environ.setdefault("TORCH_LOGS", "+torch_spyre.inductor")
os.environ.setdefault("SPYRE_INDUCTOR_LOG", "1")
os.environ.setdefault("SPYRE_INDUCTOR_LOG_LEVEL", "DEBUG")
os.environ.setdefault("SPYRE_LOG_FILE", str(PLANNER_LOG))
os.environ.setdefault("SPYRE_DUMP_COST", "1")
os.environ.setdefault("SPYRE_DUMP_COST_FILE", str(COST_FILE))

import torch  # noqa: E402
import torch_spyre  # noqa: E402
from torch_spyre._inductor import spyre_hint  # noqa: E402
from torch_spyre._inductor.wsr.propagate_named_dims import (  # noqa: E402
    declare_tensor_dim as _declare_tensor_dim,
    name_tensor_dims as _name_tensor_dims,
)


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


VERBOSE = os.environ.get("VERBOSE") == "1"
WRITE_KV = os.environ.get("WRITE_KV", "1") == "1"
BENCH_ITERS = _int("BENCH_ITERS", 0)
BENCH_REPS = _int("BENCH_REPS", 5)
SCORE_BMM_MODE = os.environ.get("SCORE_BMM_MODE", "flat")
assert SCORE_BMM_MODE in ("flat", "native", "per_stick"), SCORE_BMM_MODE
QUERY_FOLD = os.environ.get("QUERY_FOLD", "0") == "1"
QUERY_FETCH_PARTITIONS = _int("QUERY_FETCH_PARTITIONS", 1)
QUERY_BMM_WORK_DIV = os.environ.get("QUERY_BMM_WORK_DIV", "auto")
assert QUERY_BMM_WORK_DIV in ("auto", "batch_query"), QUERY_BMM_WORK_DIV
PACKED_KV_FETCH = os.environ.get("PACKED_KV_FETCH", "0") == "1"

KV_HEADS = _int("KV_HEADS", 8)
QPK = _int("QPK", 4)
HEAD_SIZE = _int("HEAD_SIZE", 128)
BLOCK_SIZE = _int("BLOCK_SIZE", 64)
NUM_BLOCKS = _int("NUM_BLOCKS", 4)
NUM_PAGES = _int("NUM_PAGES", 32)
Q_LEN = _int("Q_LEN", 1)
PAGE_GROUP = _int("PAGE_GROUP", 1)
QUERY_TOKENS = _int("QUERY_TOKENS", max(2 * Q_LEN, 8))
BATCH_REPEATS = _int("BATCH_REPEATS", 1)
PROMPT_TOKENS = _int("PROMPT_TOKENS", NUM_BLOCKS * BLOCK_SIZE)

# Fold the page's row axis into the gather's indexed axis: instead of one index
# entry selecting BLOCK_SIZE rows, use FOLD_ENTRIES entries of ROWS_PER_ENTRY
# rows each, so the entry dim has enough units to be split across cores.
# FOLD=none reproduces the original one-entry-per-page gather.
FOLD = os.environ.get("FOLD", "none")
# One index entry per slot by default: the folded source is then the natural
# slot-major view [NUM_PAGES*BLOCK_SIZE, KV_HEADS, HEAD_SIZE] and the reshape
# back to the page frame is a squeeze. Fewer entries flatten rows into the head
# axis, which makes the following permute an interleaved index that
# insert_restickify_padding rejects.
ENTRIES = _int("FOLD_ENTRIES", BLOCK_SIZE)
INDEX_2D = os.environ.get("INDEX_2D", "1") == "1"
FOLD_SRC_RANK = _int("FOLD_SRC_RANK", 3)
# GQA handling. "broadcast": one matmul over a [KV_HEADS, QPK, ...] query, so
# inductor clones the page out to the group axis -- that clone runs on
# kv x group cores while the page has no group axis, which is what bars the
# page from LX. "group": unroll the groups, so each matmul reads the page
# directly and no op reads it across an axis it does not have.
GQA = os.environ.get("GQA", "broadcast")
assert GQA in ("broadcast", "group"), GQA
# Return the per-group outputs instead of stacking them, so a wrong result can be
# attributed to the group computation or to the cat that assembles it.
RETURN_GROUPS = os.environ.get("RETURN_GROUPS") == "1"
# How each group's query slice is taken. A compiled region reads a view from
# offset 0 and ignores the strides (torch-spyre#3770), so `q[:, g]` is wrong on
# device for every g -- .contiguous() does not help, since the clone reads the
# same view. "gather" uses index_select, whose output is a real buffer at offset
# 0, which is the workaround spyre_attn.py already uses for the query rows.
Q_SLICE = os.environ.get("Q_SLICE", "heads")
assert Q_SLICE in ("heads", "gather", "slice", "slice_contig"), Q_SLICE
# Compute only the first GROUPS_COMPUTED groups, leaving the query's shape alone.
# Separates a wrong query slice at this QPK from a wrong interaction between
# several groups in one graph.
GROUPS_COMPUTED = _int("GROUPS_COMPUTED", QPK)
# Return every intermediate of block 0 / group 0 so the first divergence from CPU
# can be named instead of guessed.
DEBUG_STAGES = os.environ.get("DEBUG_STAGES") == "1"
STAGE_NAMES = ("q_g", "k_t", "v_3d", "scores_raw", "scores_masked", "probs", "tile_out")
# K cache frame. "token": [NUM_PAGES, BLOCK_SIZE, KV_HEADS, HEAD_SIZE], so the
# gathered page must be permuted to [KV_HEADS, HEAD_SIZE, BLOCK_SIZE] for q @ K^T
# -- that moves the stick dim, which a restickify does, and a buffer a restickify
# reads can never be LX ("cross-frame barrier"). "transposed": store K as
# [NUM_PAGES * KV_HEADS, HEAD_SIZE, BLOCK_SIZE], so the gather output already IS
# the matmul operand. Folding on (page, kv) gives KV_HEADS index entries, which is
# also the bmm's batch extent. Only used by GQA=group.
# "head_major" stores K in the same frame as V, [NUM_PAGES * KV_HEADS, BLOCK_SIZE,
# HEAD_SIZE], and permutes to [KV_HEADS, HEAD_SIZE, BLOCK_SIZE] in the kernel. The
# permute is a restickify, so this only reaches LX with torch-spyre#4153's local-read
# proof -- but the gather then splits kv 8 ways, matching what the restickify reads,
# and the decode store stays token-contiguous like V's.
K_LAYOUT = os.environ.get("K_LAYOUT", "token")
assert K_LAYOUT in ("token", "transposed", "head_major", "d_stick"), K_LAYOUT
# Cap cores for the attention compile only, by mutating the config the work-division
# pass reads, rather than the process-wide SENCORES env var. The pass calls
# _validate_max_cores() per compile, so this should scope the cap to this graph and
# leave the rest of the model on all 32 cores.
SENCORES_ATTN = _int("SENCORES_ATTN", 0)
# V cache frame. "token": [NUM_PAGES, BLOCK_SIZE, KV_HEADS, HEAD_SIZE], needing a
# permute to [KV_HEADS, BLOCK_SIZE, HEAD_SIZE] for probs @ V, and gathered one page
# per entry so the gather is single-core. "head_major": stored as
# [NUM_PAGES * KV_HEADS, BLOCK_SIZE, HEAD_SIZE] and gathered on (page, kv), so the
# gather output already is the matmul operand and its split lands on kv -- which is
# an output axis of probs @ V, so a consumer can match it. Only used by GQA=group.
V_LAYOUT = os.environ.get("V_LAYOUT", "token")
assert V_LAYOUT in ("token", "head_major", "d_stick"), V_LAYOUT
assert FOLD in ("none", "k", "kv"), FOLD
assert BLOCK_SIZE % ENTRIES == 0, (BLOCK_SIZE, ENTRIES)
ROWS_PER_ENTRY = BLOCK_SIZE // ENTRIES

NUM_HEADS = KV_HEADS * QPK
SCALE = HEAD_SIZE**-0.5
FP16_MIN = torch.finfo(torch.float16).min
INT32_ELEMS_PER_STICK = 32  # 128-byte stick / 4 bytes

torch_spyre._autoload()
torch.spyre.set_device(0)
torch.zeros(1, dtype=torch.float16).to("spyre")

from torch_spyre._C import (  # noqa: E402
    SpyreTensorLayout,
    get_device_dtype,
    get_elem_in_stick,
)

ELEMENTS_PER_STICK = get_elem_in_stick(torch.float16)
assert HEAD_SIZE % ELEMENTS_PER_STICK == 0, (HEAD_SIZE, ELEMENTS_PER_STICK)
D_STICKS = HEAD_SIZE // ELEMENTS_PER_STICK
assert QUERY_FETCH_PARTITIONS >= 1
assert Q_LEN % QUERY_FETCH_PARTITIONS == 0, (Q_LEN, QUERY_FETCH_PARTITIONS)
if QUERY_FETCH_PARTITIONS != 1:
    assert K_LAYOUT == "transposed", K_LAYOUT
    assert V_LAYOUT == "head_major", V_LAYOUT
    assert PAGE_GROUP == 1, PAGE_GROUP
if PACKED_KV_FETCH:
    assert QUERY_FETCH_PARTITIONS == 1
    assert K_LAYOUT == "transposed", K_LAYOUT
    assert V_LAYOUT == "head_major", V_LAYOUT
assert NUM_BLOCKS % PAGE_GROUP == 0, (NUM_BLOCKS, PAGE_GROUP)
if PAGE_GROUP != 1:
    assert K_LAYOUT == V_LAYOUT and K_LAYOUT in ("head_major", "d_stick"), (
        K_LAYOUT,
        V_LAYOUT,
    )
NUM_TILES = NUM_BLOCKS // PAGE_GROUP


def fetch_page_folded(pages, entry_index):
    """One page, gathered as ENTRIES row-blocks instead of one whole-page entry.

    `pages` is viewed as [NUM_PAGES * ENTRIES, ROWS_PER_ENTRY, KV_HEADS, HEAD_SIZE]
    so the indexed axis carries ENTRIES units per page. With INDEX_2D the index is
    [ENTRIES, 1] and gathered by advanced indexing, which keeps the entry variable
    off the index's stick axis; otherwise a 1-D `index_select` is used and the
    entry dim only splits in whole 32-entry int32 sticks.

    Returns the page in its original [BLOCK_SIZE, KV_HEADS, HEAD_SIZE] frame.
    """
    if FOLD_SRC_RANK == 3:
        # Rows and heads flattened together, so a 2-D index gives a rank-4 gather
        # output. A rank-4 source (rank-5 output) makes the relayout that reads it
        # fail in dxp: "Could not find any suitable dimension mapping".
        folded = pages.view(NUM_PAGES * ENTRIES, ROWS_PER_ENTRY * KV_HEADS, HEAD_SIZE)
    else:
        folded = pages.view(NUM_PAGES * ENTRIES, ROWS_PER_ENTRY, KV_HEADS, HEAD_SIZE)
    if INDEX_2D:
        rows = folded[entry_index]
    else:
        rows = folded.index_select(0, entry_index.reshape(-1))
    return rows.reshape(BLOCK_SIZE, KV_HEADS, HEAD_SIZE)


def paged_attn_kernel(
    query,
    query_row_index,
    k_pages,
    v_pages,
    page_index_table,
    mask_tiles,
    entry_index_table,
    group_indices=None,
    kv_index_table=None,
):
    """Online-softmax attention over NUM_BLOCKS pages for one sequence.

    Shapes:
        query             [num_tokens, NUM_HEADS, HEAD_SIZE], the whole batch's query
        query_row_index   int32; its first Q_LEN entries are this sequence's rows
        k_pages, v_pages  [NUM_PAGES, BLOCK_SIZE, KV_HEADS, HEAD_SIZE]
        page_index_table  [NUM_BLOCKS, INT32_ELEMS_PER_STICK] int32, page index in col 0
        mask_tiles        NUM_BLOCKS additive tiles of [Q_LEN, BLOCK_SIZE]
        entry_index_table one [ENTRIES, 1] int32 row-block index per block, for FOLD
        kv_index_table    one [KV_HEADS, 1] int32 (page, kv) index per block

    The index tables are lists, not one tensor sliced per block: an index tensor
    reaches the hardware as a real tensor argument, so a per-block slice's nonzero
    storage offset is dropped and every block gathers block 0's rows
    (torch-spyre#3770). page_index_table survives being sliced only because it is
    padded to INT32_ELEMS_PER_STICK, which makes each row stick-aligned.

    NUM_BLOCKS and Q_LEN are module-level constants, so Dynamo unrolls the loop.
    """
    # Gathered, not sliced: a compiled region reads a view from offset 0.
    q_rows = query.index_select(0, query_row_index[:Q_LEN])
    q = q_rows.unsqueeze(0).transpose(1, 2).reshape(KV_HEADS, QPK, Q_LEN, HEAD_SIZE)

    tile_max = None
    tile_sum = None
    tile_output = None

    for i in range(NUM_BLOCKS):
        # index_select, not `k_pages[page_idx]`: subscripting lowers to aten.index,
        # which upcasts the int32 index to int64.
        page_idx = page_index_table[i, 0:1]
        if FOLD in ("k", "kv"):
            k_page = fetch_page_folded(k_pages, entry_index_table[i])
        else:
            k_page = k_pages.index_select(0, page_idx).squeeze(0)
        if FOLD == "kv":
            v_page = fetch_page_folded(v_pages, entry_index_table[i])
        else:
            v_page = v_pages.index_select(0, page_idx).squeeze(0)
        # Token-major page to head-major for the matmuls; permutes on device.
        k_page_4d = k_page.permute(1, 0, 2).unsqueeze(1)
        v_page_4d = v_page.permute(1, 0, 2).unsqueeze(1)

        scores = torch.matmul(q, k_page_4d.transpose(-2, -1)) * SCALE
        scores = scores + mask_tiles[i]
        scores_max = torch.amax(scores, dim=-1, keepdim=True)

        if i == 0:
            tile_max = scores_max
            tile_probs = torch.exp(scores - tile_max)
            tile_output = torch.matmul(tile_probs, v_page_4d)
            tile_sum = tile_probs.sum(dim=-1, keepdim=True)
        else:
            assert tile_max is not None and tile_sum is not None and tile_output is not None
            new_max = torch.maximum(tile_max, scores_max)
            rescale = torch.exp(tile_max - new_max)
            tile_output = tile_output * rescale
            tile_sum = tile_sum * rescale
            tile_probs = torch.exp(scores - new_max)
            tile_output += torch.matmul(tile_probs, v_page_4d)
            tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)
            tile_max = new_max

    assert tile_output is not None and tile_sum is not None
    attn = tile_output / tile_sum
    attn = attn.reshape(1, NUM_HEADS, Q_LEN, HEAD_SIZE).transpose(1, 2)
    return attn.reshape(Q_LEN, NUM_HEADS, HEAD_SIZE)


def paged_attn_kernel_group_loop(
    query,
    query_row_index,
    k_pages,
    v_pages,
    page_index_table,
    mask_tiles,
    entry_index_table,
    group_indices,
    kv_index_table,
):
    """As :func:`paged_attn_kernel`, with the query groups unrolled.

    Each group runs its own pair of matmuls against the page as gathered, so the
    page is never expanded to a group axis and no consumer reads it across one.
    The online-softmax state becomes one entry per group.
    """
    if Q_SLICE == "heads":
        q = None  # taken per group from `query` directly, see below
    else:
        q_rows = query.index_select(0, query_row_index[:Q_LEN])
        q = q_rows.unsqueeze(0).transpose(1, 2).reshape(KV_HEADS, QPK, Q_LEN, HEAD_SIZE)

    tile_max: list = [None] * QPK
    tile_sum: list = [None] * QPK
    tile_out: list = [None] * QPK

    for i in range(NUM_TILES):
        tile_tokens = PAGE_GROUP * BLOCK_SIZE
        page_idx = page_index_table[i, 0:1]
        if PACKED_KV_FETCH:
            packed_page = k_pages[kv_index_table[i]].reshape(
                KV_HEADS, 2, tile_tokens, HEAD_SIZE
            )
            k_page = packed_page[:, 0]
        elif K_LAYOUT == "transposed":
            k_rows = (
                KV_HEADS * QUERY_FETCH_PARTITIONS
                if QUERY_FETCH_PARTITIONS != 1
                else KV_HEADS
            )
            k_page = k_pages[kv_index_table[i]].reshape(
                k_rows, HEAD_SIZE, BLOCK_SIZE
            )
        elif K_LAYOUT == "d_stick":
            k_page = k_pages[kv_index_table[i]].reshape(
                KV_HEADS * D_STICKS, tile_tokens, ELEMENTS_PER_STICK
            )
        elif K_LAYOUT == "head_major":
            k_page = k_pages[kv_index_table[i]].reshape(
                KV_HEADS, tile_tokens, HEAD_SIZE
            )
        elif FOLD in ("k", "kv"):
            k_page = fetch_page_folded(k_pages, entry_index_table[i])
        else:
            k_page = k_pages.index_select(0, page_idx).squeeze(0)
        if PACKED_KV_FETCH:
            v_page = packed_page[:, 1]
        elif V_LAYOUT in ("head_major", "d_stick"):
            v_page = None  # v_3d taken straight from the gather below
        elif FOLD == "kv":
            v_page = fetch_page_folded(v_pages, entry_index_table[i])
        else:
            v_page = v_pages.index_select(0, page_idx).squeeze(0)
        if PACKED_KV_FETCH:
            k_t = k_page.permute(0, 2, 1)
        elif K_LAYOUT == "transposed":
            k_t = k_page  # already [KV_HEADS, HEAD_SIZE, BLOCK_SIZE]
        elif K_LAYOUT == "d_stick":
            if SCORE_BMM_MODE == "native":
                k_t = k_page.reshape(KV_HEADS, tile_tokens, HEAD_SIZE).permute(
                    0, 2, 1
                )
            else:
                k_t = k_page.permute(0, 2, 1)
        elif K_LAYOUT == "head_major":
            k_t = k_page.permute(0, 2, 1)  # [KV_HEADS, HEAD_SIZE, BLOCK_SIZE]
        else:
            k_t = k_page.permute(1, 2, 0)
        if PACKED_KV_FETCH:
            v_3d = v_page
        elif V_LAYOUT == "d_stick":
            v_3d = v_pages[kv_index_table[i]].reshape(
                KV_HEADS * D_STICKS, tile_tokens, ELEMENTS_PER_STICK
            )
        elif V_LAYOUT == "head_major":
            v_rows = (
                KV_HEADS * QUERY_FETCH_PARTITIONS
                if QUERY_FETCH_PARTITIONS != 1
                else KV_HEADS
            )
            v_3d = v_pages[kv_index_table[i]].reshape(v_rows, tile_tokens, HEAD_SIZE)
        else:
            v_3d = v_page.permute(1, 0, 2)  # [KV_HEADS, BLOCK_SIZE, HEAD_SIZE]

        for g in range(GROUPS_COMPUTED):
            if Q_SLICE == "heads":
                # This group's heads straight off the argument, then its rows. Both
                # gathers read `query` itself; index_select on a reshaped view of a
                # computed buffer is wrong on device.
                heads = query.index_select(1, group_indices[g])
                q_g = (
                    heads.index_select(0, query_row_index[:Q_LEN])
                    .transpose(0, 1)
                    .reshape(KV_HEADS, Q_LEN, HEAD_SIZE)
                )
            elif Q_SLICE == "gather":
                q_g = q.index_select(1, group_indices[g]).reshape(KV_HEADS, Q_LEN, HEAD_SIZE)
            elif Q_SLICE == "slice_contig":
                q_g = q[:, g].contiguous()
            else:
                q_g = q[:, g]  # [KV_HEADS, Q_LEN, HEAD_SIZE]
            if K_LAYOUT == "d_stick":
                if SCORE_BMM_MODE == "native":
                    scores_raw = torch.matmul(q_g, k_t) * SCALE
                else:
                    q_sticks = q_g.reshape(
                        KV_HEADS * D_STICKS, Q_LEN, ELEMENTS_PER_STICK
                    )
                if SCORE_BMM_MODE == "per_stick":
                    q_by_stick = q_sticks.reshape(
                        KV_HEADS, D_STICKS, Q_LEN, ELEMENTS_PER_STICK
                    )
                    k_by_stick = k_t.reshape(
                        KV_HEADS,
                        D_STICKS,
                        ELEMENTS_PER_STICK,
                        tile_tokens,
                    )
                    score_partials = torch.stack(
                        [
                            torch.matmul(q_by_stick[:, stick], k_by_stick[:, stick])
                            for stick in range(D_STICKS)
                        ],
                        dim=1,
                    )
                elif SCORE_BMM_MODE == "flat":
                    score_partials = torch.matmul(q_sticks, k_t).reshape(
                        KV_HEADS, D_STICKS, Q_LEN, tile_tokens
                    )
                if SCORE_BMM_MODE != "native":
                    scores_raw = score_partials.sum(dim=1) * SCALE
            else:
                if QUERY_FETCH_PARTITIONS != 1:
                    query_partition_len = Q_LEN // QUERY_FETCH_PARTITIONS
                    q_partitioned = (
                        q_g.reshape(
                            KV_HEADS,
                            QUERY_FETCH_PARTITIONS,
                            query_partition_len,
                            HEAD_SIZE,
                        )
                        .permute(1, 0, 2, 3)
                        .reshape(
                            KV_HEADS * QUERY_FETCH_PARTITIONS,
                            query_partition_len,
                            HEAD_SIZE,
                        )
                    )
                    scores_raw = torch.matmul(q_partitioned, k_t) * SCALE
                elif QUERY_FOLD:
                    q_folded = q_g.reshape(KV_HEADS * Q_LEN, 1, HEAD_SIZE)
                    k_folded = (
                        k_t.unsqueeze(1)
                        .expand(KV_HEADS, Q_LEN, HEAD_SIZE, tile_tokens)
                        .reshape(KV_HEADS * Q_LEN, HEAD_SIZE, tile_tokens)
                    )
                    scores_raw = torch.matmul(q_folded, k_folded).reshape(
                        KV_HEADS, Q_LEN, tile_tokens
                    ) * SCALE
                else:
                    if QUERY_BMM_WORK_DIV == "batch_query":
                        with spyre_hint(
                            named_dims=["KV", "Q", "T"],
                            work_div={"KV": 8, "Q": 4},
                        ):
                            scores_raw = torch.matmul(q_g, k_t) * SCALE
                    else:
                        scores_raw = torch.matmul(q_g, k_t) * SCALE
            if QUERY_FETCH_PARTITIONS != 1:
                scores = scores_raw + mask_tiles[i]
            else:
                scores = scores_raw + mask_tiles[i]
            scores_max = torch.amax(scores, dim=-1, keepdim=True)
            if i == 0:
                tile_max[g] = scores_max
                probs = torch.exp(scores - scores_max)
                if V_LAYOUT == "d_stick":
                    probs_sticks = probs.unsqueeze(1).expand(
                        KV_HEADS, D_STICKS, Q_LEN, tile_tokens
                    )
                    tile_out[g] = torch.matmul(
                        probs_sticks.reshape(
                            KV_HEADS * D_STICKS, Q_LEN, tile_tokens
                        ),
                        v_3d,
                    ).reshape(KV_HEADS, Q_LEN, HEAD_SIZE)
                else:
                    if QUERY_FETCH_PARTITIONS != 1:
                        tile_out[g] = torch.matmul(probs, v_3d)
                    elif QUERY_FOLD:
                        probs_folded = probs.reshape(
                            KV_HEADS * Q_LEN, 1, tile_tokens
                        )
                        v_folded = (
                            v_3d.unsqueeze(1)
                            .expand(KV_HEADS, Q_LEN, tile_tokens, HEAD_SIZE)
                            .reshape(KV_HEADS * Q_LEN, tile_tokens, HEAD_SIZE)
                        )
                        tile_out[g] = torch.matmul(
                            probs_folded, v_folded
                        ).reshape(KV_HEADS, Q_LEN, HEAD_SIZE)
                    else:
                        if QUERY_BMM_WORK_DIV == "batch_query":
                            with spyre_hint(
                                named_dims=["KV", "Q", "D"],
                                work_div={"KV": 8, "Q": 4},
                            ):
                                tile_out[g] = torch.matmul(probs, v_3d)
                        else:
                            tile_out[g] = torch.matmul(probs, v_3d)
                tile_sum[g] = probs.sum(dim=-1, keepdim=True)
                if DEBUG_STAGES and g == 0:
                    return (q_g, k_t, v_3d, scores_raw, scores, probs, tile_out[0])
            else:
                new_max = torch.maximum(tile_max[g], scores_max)
                rescale = torch.exp(tile_max[g] - new_max)
                tile_out[g] = tile_out[g] * rescale
                tile_sum[g] = tile_sum[g] * rescale
                probs = torch.exp(scores - new_max)
                if V_LAYOUT == "d_stick":
                    probs_sticks = probs.unsqueeze(1).expand(
                        KV_HEADS, D_STICKS, Q_LEN, tile_tokens
                    )
                    value_out = torch.matmul(
                        probs_sticks.reshape(
                            KV_HEADS * D_STICKS, Q_LEN, tile_tokens
                        ),
                        v_3d,
                    ).reshape(KV_HEADS, Q_LEN, HEAD_SIZE)
                else:
                    if QUERY_FETCH_PARTITIONS != 1:
                        value_out = torch.matmul(probs, v_3d)
                    elif QUERY_FOLD:
                        probs_folded = probs.reshape(
                            KV_HEADS * Q_LEN, 1, tile_tokens
                        )
                        v_folded = (
                            v_3d.unsqueeze(1)
                            .expand(KV_HEADS, Q_LEN, tile_tokens, HEAD_SIZE)
                            .reshape(KV_HEADS * Q_LEN, tile_tokens, HEAD_SIZE)
                        )
                        value_out = torch.matmul(
                            probs_folded, v_folded
                        ).reshape(KV_HEADS, Q_LEN, HEAD_SIZE)
                    else:
                        if QUERY_BMM_WORK_DIV == "batch_query":
                            with spyre_hint(
                                named_dims=["KV", "Q", "D"],
                                work_div={"KV": 8, "Q": 4},
                            ):
                                value_out = torch.matmul(probs, v_3d)
                        else:
                            value_out = torch.matmul(probs, v_3d)
                tile_out[g] = tile_out[g] + value_out
                tile_sum[g] = tile_sum[g] + probs.sum(dim=-1, keepdim=True)
                tile_max[g] = new_max

    groups = [tile_out[g] / tile_sum[g] for g in range(GROUPS_COMPUTED)]
    if QUERY_FETCH_PARTITIONS != 1:
        query_partition_len = Q_LEN // QUERY_FETCH_PARTITIONS
        groups = [
            group.reshape(
                QUERY_FETCH_PARTITIONS,
                KV_HEADS,
                query_partition_len,
                HEAD_SIZE,
            )
            .permute(1, 0, 2, 3)
            .reshape(KV_HEADS, Q_LEN, HEAD_SIZE)
            for group in groups
        ]
    if RETURN_GROUPS:
        return tuple(groups)
    assert GROUPS_COMPUTED == QPK, "stacking needs every group; use RETURN_GROUPS=1"
    attn = torch.stack(groups, dim=1)
    attn = attn.reshape(1, NUM_HEADS, Q_LEN, HEAD_SIZE).transpose(1, 2)
    return attn.reshape(Q_LEN, NUM_HEADS, HEAD_SIZE)


ATTN_KERNEL = paged_attn_kernel_group_loop if GQA == "group" else paged_attn_kernel


def reshape_and_cache_kernel(key, value, k_slots, v_slots, slot_mapping):
    k_slots.index_copy_(0, slot_mapping, key)
    v_slots.index_copy_(0, slot_mapping, value)


def rows_outermost_layout(rows: int, mid: int, inner: int):
    """Row-axis-outermost layout, so a gather's indexed axis is at device position 0."""
    eps = get_elem_in_stick(torch.float16)
    sticks = (inner + eps - 1) // eps
    return SpyreTensorLayout(
        device_size=[rows, mid, sticks, eps],
        stride_map=[mid * inner, inner, eps, 1],
        device_dtype=get_device_dtype(torch.float16),
    )


def packed_rows_outermost_layout(rows: int, variants: int, mid: int, inner: int):
    eps = get_elem_in_stick(torch.float16)
    sticks = (inner + eps - 1) // eps
    return SpyreTensorLayout(
        device_size=[rows, variants, mid, sticks, eps],
        stride_map=[variants * mid * inner, mid * inner, inner, eps, 1],
        device_dtype=get_device_dtype(torch.float16),
    )


def slot_major_layout(num_slots: int):
    """Slot-axis-outermost page layout, so the slot index stays on one device dim."""
    eps = get_elem_in_stick(torch.float16)
    sticks = (HEAD_SIZE + eps - 1) // eps
    return SpyreTensorLayout(
        device_size=[num_slots, KV_HEADS, sticks, eps],
        stride_map=[KV_HEADS * sticks * eps, sticks * eps, eps, 1],
        device_dtype=get_device_dtype(torch.float16),
    )


def make_pages() -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.randn(NUM_PAGES, BLOCK_SIZE, KV_HEADS, HEAD_SIZE, dtype=torch.float16)
    v = torch.randn(NUM_PAGES, BLOCK_SIZE, KV_HEADS, HEAD_SIZE, dtype=torch.float16)
    layout = slot_major_layout(NUM_PAGES * BLOCK_SIZE)
    if PACKED_KV_FETCH:
        packed = (
            torch.stack((k, v), dim=3)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
            .reshape(NUM_PAGES * KV_HEADS, 2, BLOCK_SIZE, HEAD_SIZE)
        )
        packed_dev = packed.to(
            "spyre",
            device_layout=packed_rows_outermost_layout(
                NUM_PAGES * KV_HEADS, 2, BLOCK_SIZE, HEAD_SIZE
            ),
        )
        return packed_dev, packed_dev
    if V_LAYOUT == "d_stick":
        v_ds = (
            v.view(NUM_PAGES, BLOCK_SIZE, KV_HEADS, D_STICKS, ELEMENTS_PER_STICK)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
            .reshape(
                NUM_PAGES * KV_HEADS * D_STICKS,
                BLOCK_SIZE,
                ELEMENTS_PER_STICK,
            )
        )
        v_dev = v_ds.to(
            "spyre",
            device_layout=rows_outermost_layout(
                NUM_PAGES * KV_HEADS * D_STICKS,
                BLOCK_SIZE,
                ELEMENTS_PER_STICK,
            ),
        )
    elif V_LAYOUT == "head_major":
        v_hm = (
            v.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(NUM_PAGES * KV_HEADS, BLOCK_SIZE, HEAD_SIZE)
        )
        v_dev = v_hm.to(
            "spyre",
            device_layout=rows_outermost_layout(
                NUM_PAGES * KV_HEADS, BLOCK_SIZE, HEAD_SIZE
            ),
        )
    else:
        v_dev = v.to("spyre", device_layout=layout)
    if K_LAYOUT == "d_stick":
        k_ds = (
            k.view(NUM_PAGES, BLOCK_SIZE, KV_HEADS, D_STICKS, ELEMENTS_PER_STICK)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
            .reshape(
                NUM_PAGES * KV_HEADS * D_STICKS,
                BLOCK_SIZE,
                ELEMENTS_PER_STICK,
            )
        )
        return (
            k_ds.to(
                "spyre",
                device_layout=rows_outermost_layout(
                    NUM_PAGES * KV_HEADS * D_STICKS,
                    BLOCK_SIZE,
                    ELEMENTS_PER_STICK,
                ),
            ),
            v_dev,
        )
    if K_LAYOUT == "head_major":
        k_hm = (
            k.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(NUM_PAGES * KV_HEADS, BLOCK_SIZE, HEAD_SIZE)
        )
        return (
            k_hm.to(
                "spyre",
                device_layout=rows_outermost_layout(
                    NUM_PAGES * KV_HEADS, BLOCK_SIZE, HEAD_SIZE
                ),
            ),
            v_dev,
        )
    if K_LAYOUT == "transposed":
        # Materialised, not viewed: the gather's source must be a real tensor whose
        # indexed axis is at device position 0.
        k_t = (
            k.permute(0, 2, 3, 1)
            .contiguous()
            .reshape(NUM_PAGES * KV_HEADS, HEAD_SIZE, BLOCK_SIZE)
        )
        k_layout = rows_outermost_layout(NUM_PAGES * KV_HEADS, HEAD_SIZE, BLOCK_SIZE)
        return k_t.to("spyre", device_layout=k_layout), v_dev
    return k.to("spyre", device_layout=layout), v_dev


def write_kv(k_dev: torch.Tensor, v_dev: torch.Tensor) -> None:
    """The indirect store into slot-major views, as its own compiled graph. Writes
    Q_LEN tokens at the head of page 0, which the attention kernel then reads."""
    k_slots = k_dev.view(-1, KV_HEADS, HEAD_SIZE)
    v_slots = v_dev.view(-1, KV_HEADS, HEAD_SIZE)
    key = torch.randn(Q_LEN, KV_HEADS, HEAD_SIZE, dtype=torch.float16)
    value = torch.randn(Q_LEN, KV_HEADS, HEAD_SIZE, dtype=torch.float16)
    slots = torch.arange(Q_LEN, dtype=torch.int32)

    torch.compile(reshape_and_cache_kernel, dynamic=False)(
        key.to("spyre"), value.to("spyre"), k_slots, v_slots, slots.to("spyre")
    )

    k_err = k_dev.cpu().view(-1, KV_HEADS, HEAD_SIZE)[:Q_LEN].float() - key.float()
    v_err = v_dev.cpu().view(-1, KV_HEADS, HEAD_SIZE)[:Q_LEN].float() - value.float()
    print(
        f"KV store: k err {k_err.abs().max().item():.3e}  v err {v_err.abs().max().item():.3e}",
        flush=True,
    )


def build_attn_args(k_dev: torch.Tensor, v_dev: torch.Tensor):
    """(device args, cpu reference args) for paged_attn_kernel."""
    # More query rows than this sequence uses, so the in-graph gather selects a
    # strict subset rather than its whole source.
    num_tokens = max(QUERY_TOKENS, Q_LEN)
    query = torch.randn(num_tokens, NUM_HEADS, HEAD_SIZE, dtype=torch.float16)
    if QUERY_BMM_WORK_DIV == "batch_query":
        for name, size in (
            ("KV", KV_HEADS),
            ("Q", Q_LEN),
            ("D", HEAD_SIZE),
            ("T", PAGE_GROUP * BLOCK_SIZE),
            ("PAGE_KV", NUM_PAGES * KV_HEADS),
            ("NUM_HEADS", NUM_HEADS),
        ):
            _declare_tensor_dim(name, size)
    row_index = torch.arange(Q_LEN, dtype=torch.int32)
    page_table = torch.zeros(NUM_BLOCKS, INT32_ELEMS_PER_STICK, dtype=torch.int32)
    page_table[:, 0] = torch.arange(NUM_BLOCKS, dtype=torch.int32)
    mask_rows = Q_LEN if QUERY_FETCH_PARTITIONS == 1 else Q_LEN // QUERY_FETCH_PARTITIONS
    masks = [
        torch.zeros(mask_rows, PAGE_GROUP * BLOCK_SIZE, dtype=torch.float16)
        for _ in range(NUM_TILES)
    ]
    # Half of the last block masked off, as a real tail block would be.
    masks[-1][:, -BLOCK_SIZE // 2 :] = FP16_MIN
    # Row-block indices for the folded gather, built on the host: entry e of
    # block i is row-block page_table[i] * ENTRIES + e. Trailing size-1 axis so
    # the entry variable is not the index's stick axis.
    entry_table = [
        (page_table[i, 0] * ENTRIES + torch.arange(ENTRIES, dtype=torch.int32))
        .reshape(ENTRIES, 1)
        .contiguous()
        for i in range(NUM_BLOCKS)
    ]
    # In "heads" mode a group's index lists its NUM_HEADS-space head ids
    # (head = kv * QPK + g); otherwise it is just the group number.
    group_index_list = [
        torch.tensor(
            [kv * QPK + g for kv in range(KV_HEADS)] if Q_SLICE == "heads" else [g],
            dtype=torch.int32,
        )
        for g in range(QPK)
    ]
    # (page, kv) entries, or (page, kv, d-stick) entries for the D-stick cache.
    index_rows = KV_HEADS * D_STICKS if (
        K_LAYOUT == "d_stick" or V_LAYOUT == "d_stick"
    ) else KV_HEADS
    if PAGE_GROUP == 1:
        kv_table = [
            (
                page_table[i, 0] * index_rows
                + torch.arange(index_rows, dtype=torch.int32)
            )
            .repeat(QUERY_FETCH_PARTITIONS)
            .reshape(index_rows * QUERY_FETCH_PARTITIONS, 1)
            .contiguous()
            for i in range(NUM_BLOCKS)
        ]
    else:
        kv_table = []
        for tile in range(NUM_TILES):
            rows = []
            for kv_stick in range(index_rows):
                rows.extend(
                    int(page_table[tile * PAGE_GROUP + page, 0]) * index_rows
                    + kv_stick
                    for page in range(PAGE_GROUP)
                )
            kv_table.append(torch.tensor(rows, dtype=torch.int32).reshape(-1, 1))

    dev_args = [
        query.to("spyre"),
        row_index.to("spyre"),
        k_dev,
        v_dev,
        page_table.to("spyre"),
        [m.to("spyre") for m in masks],
        [t.to("spyre") for t in entry_table],
        [gi.to("spyre") for gi in group_index_list],
        [t.to("spyre") for t in kv_table],
    ]
    if QUERY_BMM_WORK_DIV == "batch_query":
        _name_tensor_dims(dev_args[0], ["Q", "NUM_HEADS", "D"])
        for tensor in (dev_args[2], dev_args[3]):
            _name_tensor_dims(tensor, ["PAGE_KV", "T", "D"])
    cpu_args = [
        query.float(),
        row_index,
        k_dev.cpu().float(),
        v_dev.cpu().float(),
        page_table,
        [m.float() for m in masks],
        entry_table,
        group_index_list,
        kv_table,
    ]
    return dev_args, cpu_args


def batched_attn_kernel(*args):
    outputs = [ATTN_KERNEL(*args) for _ in range(BATCH_REPEATS)]
    if BATCH_REPEATS == 1:
        return outputs[0]
    if isinstance(outputs[0], tuple):
        return tuple(item for output in outputs for item in output)
    return torch.stack(outputs)


def walk_allocations(node, acc: list) -> None:
    """Every (name, component) pair from `allocate` schedule-tree nodes."""
    if isinstance(node, dict):
        if node.get("nodeType_") == "allocate":
            acc.append((node.get("name_", "?"), node.get("component_", "?")))
        for v in node.values():
            walk_allocations(v, acc)
    elif isinstance(node, list):
        for v in node:
            walk_allocations(v, acc)


def sdsc_report() -> None:
    """Per-SDSC allocation verdicts. One line per op; the SDSC's own top-level key
    names the op, since the Tensor<N> labels are per-kernel and anonymous."""
    sdsc_root = CACHE_DIR / "inductor-spyre"
    dirs = sorted(sdsc_root.glob("*_sdsc_*")) if sdsc_root.is_dir() else []
    print(f"\n=== SDSC allocations — {len(dirs)} kernel(s) ===")
    if not dirs:
        print("no SDSC emitted: nothing compiled (a warm inductor cache?)")
        return
    for d in dirs:
        pool = ""
        bundle = d / "bundle.mlir"
        if bundle.is_file():
            m = re.search(r"device_mem_allocate (\d+) bytes", bundle.read_text())
            if m:
                pool = f"  HBM spill pool {int(m.group(1)) / 1024:.1f} KB"
        print(f"\n{d.name.split('_sdsc_')[1][:70]}{pool}")
        totals = {"lx": 0, "hbm": 0}
        for js in sorted(d.glob("sdsc_*.json"), key=lambda p: int(p.stem.split("_")[1])):
            blob = json.loads(js.read_text())
            op_name = next(iter(blob), "?")
            allocs: list = []
            walk_allocations(blob, allocs)
            lx = [n for n, c in allocs if c == "lx" or n.endswith("_lx")]
            hbm = [n for n, c in allocs if not (c == "lx" or n.endswith("_lx"))]
            totals["lx"] += len(lx)
            totals["hbm"] += len(hbm)
            spill = " HBM: " + ",".join(n.removeprefix("allocate-") for n in hbm) if hbm else ""
            print(f"  {op_name:<28} LX={len(lx)} HBM={len(hbm)}{spill}")
        print(f"  {'TOTAL':<28} LX={totals['lx']} HBM={totals['hbm']}")


ROLE_BY_KIND = {
    "index": "page gather (k_page / v_page)",
    "clone": "GQA broadcast of the page",
    "expand": "GQA broadcast of the page",
    "batched_matmul": "q @ k_page^T  and  tile_probs @ v_page",
    "exp": "tile_probs (the P operand)",
    "amax": "row max",
    "sum": "row sum",
    "index_put_": "KV cache store",
    "restickify": "relayout copy",
}


def parse_verdicts() -> list[list[tuple[str, str, str]]]:
    """Per-graph blocks of (buf, kind, verdict) from the planner log.

    `lx_pinning` fires once per op at the end of layout planning, so the lines
    arrive in contiguous runs, one run per compiled graph.
    """
    if not PLANNER_LOG.is_file():
        return []
    pat = re.compile(r"lx_pinning: (\S+) \((\S+)\) . (.*)")
    blocks: list[list[tuple[str, str, str]]] = []
    current: list[tuple[str, str, str]] = []
    for ln in PLANNER_LOG.read_text(errors="replace").splitlines():
        m = pat.search(ln)
        if m:
            current.append((m.group(1), m.group(2), m.group(3).strip()))
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def verdict_report() -> None:
    """LX vs not, aggregated by op kind, with the planner's refusal reasons."""
    blocks = parse_verdicts()
    if not blocks:
        print("\n(no planner verdicts: scratchpad.allocator did not log at DEBUG)")
        return
    for block in blocks:
        store = any(k == "index_put_" for _, k, _ in block)
        label = "KV store graph" if store else "attention graph"
        print(f"\n=== LX verdicts by op kind — {label} ({len(block)} ops) ===")
        kinds: dict[str, dict] = {}
        for _, kind, verdict in block:
            e = kinds.setdefault(kind, {"lx": 0, "no": 0, "why": []})
            if verdict == "lx":
                e["lx"] += 1
            else:
                e["no"] += 1
                short = verdict.split(":")[0] if "PerCoreView" in verdict else verdict
                if short not in e["why"]:
                    e["why"].append(short)
        for kind, e in sorted(kinds.items(), key=lambda kv: -kv[1]["no"]):
            role = ROLE_BY_KIND.get(kind, "")
            why = ("  <- " + "; ".join(e["why"])) if e["why"] else ""
            print(f"  {kind:<16} LX {e['lx']:>2}   not-LX {e['no']:>2}   {role}{why}")
        if VERBOSE:
            for buf, kind, verdict in block:
                print(f"    {buf:<8} {kind:<16} {verdict[:150]}")


def classify(shape: list[int]) -> str:
    """Name the kernel-level role of a buffer from its torch shape."""
    q, b, d, kv, h = Q_LEN, BLOCK_SIZE, HEAD_SIZE, KV_HEADS, NUM_HEADS
    roles = {
        (NUM_PAGES, b, kv, d): "page cache (arg)",
        (NUM_PAGES * b, kv, d): "page cache, slot-major view",
        (1, b, kv, d): "k_page / v_page  <- the gather",
        (ENTRIES, 1, ROWS_PER_ENTRY, kv, d): "k_page / v_page  <- the folded gather",
        (ENTRIES, ROWS_PER_ENTRY, kv, d): "k_page / v_page  <- the folded gather",
        (ENTRIES, 1, ROWS_PER_ENTRY * kv, d): "k_page / v_page  <- the folded gather",
        (ENTRIES, ROWS_PER_ENTRY * kv, d): "k_page / v_page  <- the folded gather",
        (b, kv, d): "k_page / v_page, row-major",
        (kv, QPK, d, b): "K clone, GQA-expanded + transposed for q @ K^T",
        (kv, QPK, b, d): "V clone, GQA-expanded",
        (kv, 1, b, d): "page, head-major",
        (q, h, d): "query rows",
        (1, h, d): "query rows",
        (kv, QPK, q, b): "scores / tile_probs  (the P operand)",
        (kv, QPK, q, d): "tile_output",
        (kv, QPK, q, 1): "tile_max / tile_sum",
        (kv, d, b): "k_page, head-major transposed for q @ K^T",
        (kv, 1, d, b): "k_page  <- the transposed gather",
        (kv, 1, b, d): "v_page  <- the head-major gather",
        (NUM_PAGES * kv, b, d): "V page cache, head-major (arg)",
        (NUM_PAGES * kv, d, b): "K page cache, transposed (arg)",
        (kv, b, d): "v_page, head-major",
        (kv, q, b): "scores / tile_probs  (the P operand)",
        (kv, q, d): "tile_output",
        (kv, q, 1): "tile_max / tile_sum",
    }
    role = roles.get(tuple(shape), "")
    if role and b == d and tuple(shape) == (kv, QPK, q, d):
        # BLOCK_SIZE == HEAD_SIZE makes the scores and tile_output shapes identical.
        return "scores / tile_probs / tile_output (shapes collide at head_size == block_size)"
    return role


def role_report() -> None:
    """Residency by kernel-level role, joined from the cost dump's per-op lines."""
    if not COST_FILE.is_file():
        return
    pat = re.compile(r"output\s+(\S+)\s+torch \[([\d, ]+)\] -> device \[[\d, ]+\] in (LX|HBM)")
    rows = []
    for ln in COST_FILE.read_text(errors="replace").splitlines():
        m = pat.search(ln)
        if m:
            shape = [int(x) for x in m.group(2).split(",")]
            rows.append((m.group(1), shape, m.group(3), classify(shape)))
    print(f"\n=== residency by role ({len(rows)} produced buffers) ===")
    for op, shape, where, role in rows:
        if role or VERBOSE:
            print(f"  {where:<4} {op:<8} {str(shape):<22} {role}")
    named = [r for r in rows if r[3]]
    print(f"\n  LX  roles: {', '.join(sorted({r[3] for r in named if r[2] == 'LX'})) or '-'}")
    print(f"  HBM roles: {', '.join(sorted({r[3] for r in named if r[2] == 'HBM'})) or '-'}")


def store_report() -> None:
    """Mutation relayout copies torch-spyre inserted for the store's destination."""
    if not PLANNER_LOG.is_file():
        return
    copies = [
        ln
        for ln in PLANNER_LOG.read_text(errors="replace").splitlines()
        if "mutation relayout copy" in ln
    ]
    print(f"\n=== KV store relayout copies ({len(copies)}) ===")
    for ln in copies:
        print("  " + ln.split("]", 2)[-1].strip())


print(
    f"blocks={NUM_BLOCKS} q_len={Q_LEN} kv_heads={KV_HEADS} qpk={QPK} "
    f"query_tokens={QUERY_TOKENS} "
    f"batch_repeats={BATCH_REPEATS} "
    f"prompt_tokens={PROMPT_TOKENS} "
    f"head_size={HEAD_SIZE} block_size={BLOCK_SIZE} "
    f"write_kv={WRITE_KV} solver={os.environ.get('LAYOUT_SOLVER', 'default')} "
    f"fold={FOLD} entries={ENTRIES} rows_per_entry={ROWS_PER_ENTRY} index_2d={INDEX_2D} "
    f"fold_src_rank={FOLD_SRC_RANK} gqa={GQA} q_slice={Q_SLICE} "
    f"groups_computed={GROUPS_COMPUTED} k_layout={K_LAYOUT} "
    f"sencores_attn={SENCORES_ATTN or 'off'} v_layout={V_LAYOUT} "
    f"d_sticks={D_STICKS} bench_iters={BENCH_ITERS} "
    f"score_bmm_mode={SCORE_BMM_MODE} "
    f"query_fold={QUERY_FOLD} "
    f"query_fetch_partitions={QUERY_FETCH_PARTITIONS} "
    f"query_bmm_work_div={QUERY_BMM_WORK_DIV} "
    f"packed_kv_fetch={PACKED_KV_FETCH} "
    f"page_group={PAGE_GROUP} tiles={NUM_TILES} "
    f"fix4258={os.environ.get('SPYRE_FIX_4258', '0')}",
    flush=True,
)
print(f"artifacts: {OUT}", flush=True)

k_dev, v_dev = make_pages()
if WRITE_KV:
    write_kv(k_dev, v_dev)

dev_args, cpu_args = build_attn_args(k_dev, v_dev)
# Independent check that the two kernel shapes agree on CPU, so a restructuring
# bug cannot hide inside the device-vs-CPU comparison of a single closure.
if not RETURN_GROUPS and K_LAYOUT == "token" and V_LAYOUT == "token":
    xcheck = (paged_attn_kernel(*cpu_args) - paged_attn_kernel_group_loop(*cpu_args)).abs()
    print(f"cpu cross-check broadcast vs group-loop: max abs diff {xcheck.max().item():.3e}")

if SENCORES_ATTN:
    from torch_spyre._inductor import config as _ts_config

    print(f"sencores {_ts_config.sencores} -> {SENCORES_ATTN} for the attention compile")
    _ts_config.sencores = SENCORES_ATTN

compiled_attn = torch.compile(batched_attn_kernel, dynamic=False)
raw = compiled_attn(*dev_args)
ref_raw = batched_attn_kernel(*cpu_args)
if DEBUG_STAGES:
    for name, dev_t, cpu_t in zip(STAGE_NAMES, raw, ref_raw):
        d = (dev_t.cpu().float() - cpu_t).abs().max().item()
        scale = cpu_t.abs().max().item() or 1.0
        flag = "OK  " if d / scale < 2e-2 else "WRONG"
        print(f"  {flag} {name:<14} {str(tuple(dev_t.shape)):<20} max abs {d:.3e}  rel {d / scale:.3e}")
    raise SystemExit(0)
if isinstance(raw, tuple):
    print("per-group max abs diff (RETURN_GROUPS): " + "  ".join(
        f"g{g}={(raw[g].cpu().float() - ref_raw[g]).abs().max().item():.3e}"
        for g in range(len(raw))
    ))
    got = torch.cat([t.cpu().float().reshape(-1) for t in raw])
    ref = torch.cat([t.reshape(-1) for t in ref_raw])
else:
    got = raw.cpu().float()
    ref = ref_raw

# Characterise a mismatch: an all-zero or constant `got` points at a dropped
# write (a cat destination, say) rather than an arithmetic or layout error.
print(f"got  absmax {got.abs().max().item():.4e}  mean {got.mean().item():+.4e}  zeros {(got == 0).sum().item()}/{got.numel()}")
print(f"ref  absmax {ref.abs().max().item():.4e}  mean {ref.mean().item():+.4e}")
if got.dim() == 3:
    per_head = (got - ref).abs().amax(dim=(0, 2))
    print("per-head max abs diff: " + " ".join(f"{v:.2e}" for v in per_head.tolist()))
    # Is `got` the right values in group-major head order? Output head h is
    # (kv=h//QPK, g=h%QPK); a group-major assembly puts it at g*KV_HEADS+kv.
    perm = torch.tensor(
        [(h % QPK) * KV_HEADS + (h // QPK) for h in range(NUM_HEADS)], dtype=torch.long
    )
    d_perm = (got - ref.index_select(1, perm)).abs().max().item()
    print(f"max abs diff vs group-major-reordered ref: {d_perm:.3e}")
    # Which reference head does each computed head actually match? A clean
    # permutation means the arithmetic is right and only the assembly is wrong;
    # no match anywhere means the values themselves are wrong.
    g2 = got[0].float()
    r2 = ref[0].float()
    dist = (g2.unsqueeze(1) - r2.unsqueeze(0)).abs().amax(dim=-1)
    best = dist.argmin(dim=1)
    resid = dist.gather(1, best.unsqueeze(1)).squeeze(1)
    print("got head -> best ref head: " + " ".join(
        f"{h}->{best[h].item()}{'' if resid[h] < 1e-2 else '?'}" for h in range(NUM_HEADS)
    ))
    print(f"heads with a clean match (<1e-2): {(resid < 1e-2).sum().item()}/{NUM_HEADS}")

diff = (got - ref).abs().max().item()
denom = ref.abs().max().item() or 1.0
print(f"\nattention: max abs diff vs CPU {diff:.3e}  (rel {diff / denom:.3e})")
print("NUMERICS OK" if diff / denom < 2e-2 else "NUMERICS SUSPECT — do not trust the layout")

if BENCH_ITERS:
    for _ in range(3):
        compiled_attn(*dev_args)
    raw[0].cpu() if isinstance(raw, tuple) else raw.cpu()

    def timed_batch(count: int) -> float:
        start = time.perf_counter()
        result = None
        for _ in range(count):
            result = compiled_attn(*dev_args)
        result[0].cpu() if isinstance(result, tuple) else result.cpu()
        return time.perf_counter() - start

    samples = []
    for _ in range(BENCH_REPS):
        one = timed_batch(BENCH_ITERS)
        two = timed_batch(2 * BENCH_ITERS)
        samples.append((two - one) / BENCH_ITERS)
    median_us = statistics.median(samples) * 1e6
    print(
        f"attention benchmark: median {median_us:.1f} us/launch "
        f"({BENCH_REPS} differential reps, N={BENCH_ITERS})"
    )
    if PROMPT_TOKENS > Q_LEN:
        chunks = (PROMPT_TOKENS + Q_LEN - 1) // Q_LEN
        estimated_ms = median_us * chunks / 1000
        print(
            f"chunked prompt estimate: {PROMPT_TOKENS} tokens / {Q_LEN} "
            f"per chunk = {chunks} launches, {estimated_ms:.3f} ms"
        )

sdsc_report()
verdict_report()
role_report()
store_report()
print(f"\nartifacts: {OUT}")
