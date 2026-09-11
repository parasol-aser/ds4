# Review of the revised paper

Reviewed 2026-09-11. This is a second-pass review of `main.tex`, the 18-page
`main.pdf`, the reproduction scripts, and the packaged logs and results. It
supplements [REVIEW.md](REVIEW.md); the line numbers below refer to this revision.

The revision is substantially stronger, and the central speed result is supported
by the packaged measurements. I would still fix the calculation, comparison, and
wording issues below before calling it finished.

## What now checks out

- The architecture description now gives 24 query heads and 4 KV heads, and the
  DeltaNet description is more precise about normalization, scaling, and decay.
- The streaming-bound arithmetic is internally consistent: approximately
  `13.71 GB / 616 GB/s + 1.35 GB / 670 GB/s = 24.3 ms`.
- The paper discloses the different MLX quantization and smaller output head,
  llama.cpp's sampler defaults, the Q4_K greedy-history divergence, and the narrow
  scope of the downstream evaluation.
- The repeated measurements, paired perplexity analysis, and ablation plot are
  substantial improvements. The MLX perplexity scoring count is now 40,920,
  consistent with 40 chunks and the stated scored positions.
- Running `analyze.py` against the packaged logs in a temporary output directory
  reproduced `scripts/out/results.json` exactly. Running `tables.py` on that
  result reproduced every packaged generated table exactly. This establishes
  consistency between logs and tables; the analysis bugs below still matter.
- The Q4_K forward comparison reproduces at **30.6379 ms for ds4 versus
  42.2531 ms for llama.cpp**, a latency reduction of approximately **27.5%**.
  Including ds4's first decode forward changes its mean to **30.6465 ms**, so
  the exclusion issue below has very little effect on this result.
- The ablation supports approximately **34.97 ms to 30.80 ms**, or a **12%**
  reduction, with the limitations of the stated switch-based comparison.
- All 18 PDF pages were visually inspected. No unresolved references or obvious
  content extending beyond the horizontal page margins were found. Figure 1 has
  a readability problem described below.
- The shell scripts passed syntax checks, and the Python scripts passed syntax
  compilation. No model inference or large benchmark was rerun for this review.

## Remaining calculation and comparison issues

### 1. MLX's conversion still uses output tokens rather than decode forwards

Locations: [scripts/analyze.py](scripts/analyze.py), lines 55–61;
[main.tex](main.tex), lines 550–557; the MLX context table and plot.

The MLX parser currently sets `loop = 1000 / generation_tps`. However, the pinned
mlx-lm implementation resets its generation timer after obtaining the first token
and reports `(n + 1) / elapsed`. Thus its reported count includes the first output
token, while the paper defines time per subsequent decode forward. This follows
from the timer and response construction in the
[mlx-lm v0.31.3 source](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/generate.py#L674-L710).

For these runs, with 64 reported output tokens, the count conversion should be
`1000 * 64 / (63 * generation_tps)`, as already done for ds4. Prefer deriving the
count from the log and explicitly handling early termination.

The count correction alone gives these means from the existing logs:

| MLX run | Current ms | Count-corrected ms per decode forward |
| --- | ---: | ---: |
| Short, 20-token prompt | 29.3784 | 29.8448 |
| 1k context | 29.9898 | 30.4658 |
| 4k context | 31.1873 | 31.6823 |
| 8k context | 32.4160 | 32.9305 |
| 16k context | 33.5653 | 34.0981 |
| 32k context | 36.5221 | 37.1018 |

Consequently, the short-prompt prose should say approximately **29.8 ms**, and
the context range becomes approximately **30.5–37.1 ms** for the existing inputs.
Scale the standard deviations by the same factor and regenerate the figure and
tables. MLX performs asynchronous work, so count normalization should not be
described as proving identical wall-clock boundaries across engines.

### 2. The MLX context prompts differ because shell substitution strips newlines

Locations: [scripts/run_all.sh](scripts/run_all.sh), line 50;
[main.tex](main.tex), lines 783–786.

`--prompt "$(cat "$OUT/prompt_$n.txt")"` removes trailing newlines. ds4 and
llama.cpp receive the original files, but MLX receives strings missing the final
two newlines. The caption's explanation that MLX's tokenizer counts one fewer
token is therefore incorrect for these inputs.

I checked the cached MLX tokenizer at revision
`3e6447f082e89cc7f0bc6e5441afd38dfce760ff`. Encoding the original file and the
shell-stripped string reproduces the difference exactly:

| Prompt | Full file, tokens | Trailing newlines removed, tokens |
| --- | ---: | ---: |
| 1k | 1,450 | 1,449 |
| 4k | 4,306 | 4,305 |
| 8k | 8,590 | 8,589 |
| 16k | 17,158 | 17,157 |
| 32k | 32,866 | 32,865 |

Pass the prompt through standard input, for example `--prompt -` with input
redirected from the prompt file. The pinned CLI reads standard input using
`sys.stdin.read()`, preserving the file contents; see its
[prompt-loading code](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/generate.py#L1881-L1882).
Record the input hash and token count, rerun the MLX context measurements, and
update the caption. This issue does not affect the short prompt, which is passed
through the preserved `$SHORT` variable.

### 3. The negative KL divergence is a numerical calculation error

Locations: [scripts/cmp.py](scripts/cmp.py), lines 2 and 9–10;
[tables/correctness.tex](tables/correctness.tex), the short-prompt step-16 row.

The table reports **−7.2e−09** for KL. KL divergence cannot be negative in exact
arithmetic. Here the softmax normalization and logarithms are computed in
float32, which is insufficient for some of the very small reported differences.

Recomputing the saved float32 logits after conversion to float64, using stable
log-softmax normalization, gives:

| Point | Existing reported KL | Recomputed KL |
| --- | ---: | ---: |
| Short, step 16 | −7.2e−09 | +1.1481e−08 |
| Short, step 4 | 1.69e−07 | 1.7431e−07 |
| Short, step 40 | 2.53e−07 | 8.2935e−09 |
| 8k, step 4 | 4.96e−07 | 2.9034e−07 |

Use float64 for normalization, logarithms, and accumulation, then regenerate the
comparison logs and tables from the existing dumps. Do not just clamp negative
values to zero. The recomputed distributions remain close at the eight included
points; this is an error in the reported metric, not evidence of a new model
failure. All nine saved dump pairs contain 248,320 entries, and the excluded
short step-63 pair still has different argmaxes after the histories diverge.

### 4. The headline loop columns still compare different decoding settings

Locations: [scripts/tables.py](scripts/tables.py), line 24;
[main.tex](main.tex), lines 499–511.

The ds4 loop column is greedy, while llama.cpp's default and matched loop
columns are sampled. The caption now discloses this, but a reader still cannot
compare the two engines' sampled loops directly in the headline table.

The needed ds4 result is already in `results.json`. For Q4_K it is **31.0591 ms**
for the sampled loop, compared with **43.5585 ms** for llama.cpp's matched
eval-plus-sampling measurement. Add a ds4 sampled column or split the table
into greedy and matched-sampling comparisons. Preserve the separate forward
comparison, which already supports the central speed claim.

The caption also says all results are means over five runs. The llama.cpp
default-sampler column has **three runs**, as specified in `run_all.sh` and
recorded in `results.json`. State the actual repetition count for that column.

### 5. The timing labels and warm-up convention need another small clarification

Locations: [main.tex](main.tex), lines 222–242;
[scripts/analyze.py](scripts/analyze.py), lines 32 and 49–52;
[figs/timeline.tex](figs/timeline.tex), lines 21–26.

The forward timer measures host elapsed time around the forward call, including
its encoding and synchronization work. Calling it just the GPU forward can be
confused with the separate `GPUStartTime`/`GPUEndTime` stage measurements. Label
it as forward-call elapsed time and reserve GPU-span terminology for the actual
GPU timestamps.

Likewise, llama.cpp's `eval time + sampling time` is a proxy for the generation
loop; it does not include all emission and other loop overhead covered by ds4's
wall clock. Either label the proxy explicitly or instrument matching boundaries.

Finally, `ds4_greedy()` drops the first of 63 decode evaluations and averages
the remaining 62, while the llama.cpp forward average includes all 63. Use one
policy or disclose the difference. Its measured effect on the Q4_K mean is only
about **0.009 ms**, so this correction does not change the speed conclusion.

## Remaining claims and presentation

### 6. The engine-independent quality assertion still overstates the evidence

Location: [main.tex](main.tex), lines 449–451.

The claim that file quality is a property of quantization and not of the engine
does not follow from agreement at a few logit checkpoints. The paper itself
reports a greedy divergence, and the GGUF perplexity results come from
llama-perplexity rather than ds4. An implementation can change effective quality
through numerical behavior or bugs even when the weight file is identical.

Suggested replacement:

> We measure the quantization quality of the GGUF files using llama.cpp. The
> preceding logit comparisons separately assess ds4's numerical agreement with
> that implementation on the tested prompts and steps.

If engine-independent quality is central to the argument, add a ds4 perplexity
measurement on the same scored tokens instead of inferring it from spot checks.

### 7. The paired perplexity column is mislabeled

Locations: [main.tex](main.tex), lines 453–454 and 473–475;
[scripts/tables.py](scripts/tables.py), line 65.

The displayed percentage is `100 * (exp(mean_nll_difference) - 1)`, which is a
**relative perplexity change**, not a percentage difference in per-chunk loss.
Label it accordingly, or report the paired NLL difference in nats per token.

The reported standard error is currently `100 * SE(mean_nll_difference)`, a
small-change approximation on the percentage scale. For a relative-perplexity
column, use the delta-method standard error
`100 * exp(mean_nll_difference) * SE`, or transform the endpoints of an interval
on the NLL scale. State the uncertainty convention; the qualitative conclusion
does not depend on this small transformation adjustment.

### 8. Figure 1 is too small to read in the PDF

Locations: [main.tex](main.tex), line 222;
[figs/timeline.tex](figs/timeline.tex), lines 24–26; PDF page 4.

The long explanatory TikZ node has no wrapping width. It enlarges the picture's
bounding box, so resizing the entire picture to the text width reduces its
labels to approximately **3.5 pt** in the PDF.

Move that explanation into the caption or give the node an explicit text width.
Then size the diagram so its labels are readable at normal page scale. Correct
the timing labels at the same time, as described in finding 5.

## Reproduction and smaller corrections

### 9. The documented generation sequence omits `tables.py`

Locations: [main.tex](main.tex), lines 839–850;
[scripts/README.md](scripts/README.md), lines 7–9 and 21.

The instructions run `analyze.py` and `figures.py` and say analysis produces the
tables. The LaTeX table files are actually written by `tables.py`. Following the
documented sequence after new measurements can therefore leave stale table
bodies beside regenerated figures.

Add `python3 tables.py` after `python3 analyze.py`, list it in the README, and
update the comment on the analysis command. Add the final LaTeX build command
so a reader can regenerate the PDF too.

### 10. Correctness failures can be masked by `tee`

Locations: [scripts/run_correctness.sh](scripts/run_correctness.sh), lines 4 and
19; [main.tex](main.tex), lines 852–853.

`cmp.py` exits nonzero on an argmax mismatch or malformed dump, but its output
is piped through `tee` under `set -eu` without `pipefail`. A successful `tee`
masks the comparison failure. Thus the claim that the scripts stop on every
failed command is not true.

Enable pipeline failure handling and explicitly handle the intentionally
non-comparable short step-63 case. Simply enabling `pipefail` with the current
loop would stop at that known divergence and prevent the later long-prompt
checks. Unexpected errors should remain failures, and the table exclusion should
be justified from recorded token histories when rerunning rather than solely
from the hard-coded `short_s63` name in `tables.py`.

### 11. Finish pinning the inputs and aligning the published artifact layout

Locations: [scripts/run_all.sh](scripts/run_all.sh), line 20;
[scripts/run_q40.sh](scripts/run_q40.sh), line 11;
[main.tex](main.tex), lines 803–837.

The recorded model revisions and hashes are useful, but the scripts still fetch
the MLX model without a `revision=` argument and the importance matrix from
`resolve/main`. Pin those fetches to the recorded releases and verify the full
expected hashes when using existing local files. Pin or hash the evaluation
input as well.

The paper describes `paper/scripts/`, `paper/logs/`, and a root `results.json`,
whereas this checkout uses `misc/paper/scripts/out/logs/` and
`misc/paper/scripts/out/results.json`. The SHA-256 path is also described
differently. This may be an intended publication layout; align the instructions
with the actual published tree before release. The README's statement that the
root is always two levels up is also stale: the scripts now search ancestors.

The earlier DeepSeek model is explicitly acknowledged as no longer distributed.
Keep that secondary result provisional until it is rerun on the available file
or its exact input is made reproducible. The current download command alone
does not reproduce the historical measurement.

### 12. Small textual corrections and useful qualifications

- `main.tex`, line 432: change **"Two prompts and eight steps"** to **"three
  prompts and eight (prompt, step) points"**. Step 0 is a prefill checkpoint, and
  the table has three distinct prompt lengths.
- `main.tex`, line 847: `run_correctness.sh` currently computes **nine** pairs;
  eight are included after excluding the divergent-history pair. The README's
  step list also omits step 40.
- `main.tex`, line 195: **"Norms and biases add 20 KB"** is too broad. The output
  norm is about 20 KB; layer norms and biases are already part of the layer
  tensor accounting. Say **"The output norm adds 20 KB"**.
- Where the introduction says the engines sample the same distribution, use
  **"the same sampling settings"**. Matching sampler parameters alone does not
  guarantee identical distributions along independently generated histories.
- Keep the causal explanations for the residual gap to llama.cpp qualified.
  The measured 4.2 ms switch ablation supports that particular optimization
  increment. Attributing the rest chiefly to dispatch count and encoding is an
  explanation supported by profiles, rather than an isolated ablation of those
  factors across engines.
- Cite the specific Qwen model/configuration or report for the architecture,
  alongside the pinned weight releases, instead of relying on an organization
  landing page for those details.

## Scope and suggested finishing order

First fix the MLX count conversion and float64 KL calculation, then regenerate
the affected analysis, tables, and figures. Add the existing sampled ds4 result
to the headline comparison, correct the quality and timing wording, and repair
Figure 1. Preserve the full MLX context prompts and rerun those measurements
before presenting them as using the same inputs. Finish the reproduction
sequence and failure handling, then inspect the rebuilt PDF.

This review changes no paper source, benchmark scripts, or recorded results.
Validation used the saved logs and dumps, lightweight syntax and tokenizer
checks, temporary regeneration of the analysis and tables, and the pinned MLX
source. It does not establish that a clean machine can reproduce the full
benchmark pipeline or independently validate hardware performance.

Reviewed file SHA-256 values:

```text
c9e3fe89f17266c205da304590e9fb6b03f84d2fbadeadd0cdfd4cf3aa3b1e25  main.tex
0a34f2e8cc2e9b82e4e44150fa66eda3c9f2f0a8514a9382f5cc54d1cebc4bde  main.pdf
```
