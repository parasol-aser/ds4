**Review of “Where the Tokens Go: Decomposing and Closing the Decode Gap for a 27B Hybrid LLM on Apple Silicon”**

Reviewed September 10, 2026. This records the findings and suggestions from the review of [main.tex](main.tex) and all 16 pages of [main.pdf](main.pdf).

The stage totals reproduce from the saved experiment logs, and the reported Apple clang fast-math behavior was reproduced with a small compiler experiment. The main issues are inconsistent benchmark timing definitions, unmatched baseline configurations, incorrect model and experiment labels, and causal or quality claims stronger than the evidence.

This was a source, metadata, log, and document review. The large model benchmarks were not rerun. The manuscript and implementation were not changed during the review. Line numbers below refer to the version inspected on September 10 and may move after edits.

Each item distinguishes a confirmed discrepancy, a limitation of the evidence, or a suggested improvement. A discrepancy does not by itself invalidate the measured optimization gains.

1. **Highest priority: use equivalent throughput measurements across engines.**

   Status: confirmed methodological discrepancy.

   Locations: main.tex lines 459–495; [scripts/analyze.py](scripts/analyze.py), lines 3–13; [scripts/figures.py](scripts/figures.py), lines 7–13. Relevant implementation locations at review time were ds4_cli.c lines 622–715 and ds4.c lines 54593–54642.

   The analysis takes ds4's reported generation throughput, which times the generation loop including sampling and token emission, but takes llama.cpp's “eval time” throughput, which measures evaluation separately from sampling. The pinned llama.cpp source explicitly reports sampling time and evaluation time separately in common/sampling.cpp, and accumulates evaluation time in src/llama-context.cpp.

   Therefore the sampled columns do not measure equivalent work. The explanation that llama.cpp's sampled and greedy rates coincide because top-k makes sampling cheap is not established by a metric that excludes sampling in the first place.

   Suggested change: report two clearly defined quantities for every engine: forward-only latency and complete generation-loop latency. Use the same timing boundaries, token-count convention, prompt, output length, and sampling parameters. Measure the sampling component directly before interpreting its cost. Recompute cross-engine speedup percentages after normalizing these definitions.

2. **Normalize how the first generated token is counted.**

   Status: confirmed counting difference.

   Locations: the same throughput analysis and ds4 decode loops as item 1.

   In a run producing 64 tokens, the first token is sampled from prefill logits. The ds4 generation loop can therefore count 64 generated tokens while performing 63 decode forwards. The saved llama.cpp logs report 63 evaluation runs.

   Counting 64 rather than 63 over otherwise identical elapsed time changes a rate by approximately 1.6%. This is smaller than the principal reported speedup, but matters for precise ceiling fractions and small improvements.

   Suggested change: define whether the metric is output tokens per generation-loop second, completed decode forwards per second, or inter-token latency after the first output. Use that convention consistently. A count normalization alone would multiply a 64-token ds4 rate by 63/64; it would not also correct the sampling and timing-boundary differences.

3. **Correct the attention query-head count.**

   Status: confirmed factual error.

   Location: main.tex lines 168–172, §2.2.

   The model has 24 attention query heads of dimension 256 and four KV heads. The manuscript says 16 query heads. Both the local GGUF metadata and the [official Qwen configuration](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json) specify 24.

   The attention-layer parameter count in Table 1 already appears consistent with 24 query heads, so this is primarily a prose correction.

4. **Complete the DeltaNet mechanics and distinguish decoder layers from auxiliary weights.**

   Status: description and accounting improvements.

   Locations: main.tex lines 159–191; metal/qwen35.metal lines 23–112.

   The local implementation includes SiLU after the depthwise convolution, L2 normalization of q and k, and a 1/sqrt(128) query scale. Define A = -exp(A_log) in the decay expression so that its sign and relation to checkpoint parameters are clear.

   The official configuration has 64 base decoder layers and one MTP layer. The local GGUF reports 65 blocks because it includes that auxiliary layer; this does not make the manuscript's 64-layer decoder description wrong. Explain which weights contribute to the per-token byte budget and which auxiliary weights are excluded.

   Replace an unqualified “every token reads every weight” with a statement about the active decoder matrices. The embedding lookup reads one row, and ordinary text decode does not use every auxiliary tensor in the model distribution.

5. **Use one consistent derivation of the streaming bound.**

   Status: confirmed arithmetic inconsistency.

   Locations: main.tex lines 61–65 and 195–200.

   At 15.05 GB/token, 670 GB/s implies approximately 44.5 tokens/s, not 41. Later the paper substitutes 620 GB/s to obtain approximately 41, without reconciling the introduction.

   A more explicit estimate using the paper's measured per-format rates is:

   ~~~text
   T_weights = 13.70 GB / (613 GB/s) + 1.35 GB / (670 GB/s)
             ≈ 24.36 ms/token

   R_weights ≈ 41.0 tokens/s
   ~~~

   This provides a rationale for the approximately 41 tokens/s number without an unexplained 620 GB/s assumption. Call it an estimated weight-streaming bound: applying the fastest measured kernel rate to every matrix shape is an assumption, and the bound excludes other necessary work.

   Keep the nominal 800 GB/s bound, the measured Q8_0 and Q4_K kernel rates, the estimated full-model weight-streaming bound, and the observed complete decode rate distinct.

6. **Do not treat a roofline shortfall as a measured partition of token time.**

   Status: inference stronger than the evidence.

   Locations: main.tex lines 28–35, 64–65, 98–108, 284–304, and 705–708.

   The claim that more than half of every token is something other than moving weights is derived from an ideal nominal-bandwidth calculation. It is not a direct measurement of a non-streaming time interval. A slower streaming kernel still spends time moving weights, and memory access, dequantization, and instruction execution can overlap.

   Similarly, weight bytes divided by kernel time is an effective bandwidth metric; it is not automatically a measurement of physical DRAM traffic or proof of saturation.

   Suggested change: describe the measured utilization of the chosen bound, and label additional cost attribution as estimated or inferred. Separate measured stage durations from a theoretical lower bound. Hardware counters or controlled kernel experiments would strengthen claims about physical traffic and the cause of the remaining gap.

7. **Correct the sampling defaults and recorded llama.cpp settings.**

   Status: confirmed factual errors.

   Locations: main.tex lines 362–367 and 483–490.

   The paper says the model-card default has no top-k and llama.cpp uses top-k 40. The [official generation configuration](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/generation_config.json) specifies temperature 1.0, top-p 0.95, and top-k 20.

   The saved llama.cpp sampled-run logs for Q4, Q8, and UD report:

   ~~~text
   top_k = 20
   top_p = 0.950
   min_p = 0.050
   temp = 1.000
   ~~~

   The effective run settings matter more than generic source-code defaults, since model metadata can influence them.

   Suggested change: describe ds4's unrestricted top-p setting as the experimental configuration chosen for the study, and explicitly set all sampling parameters for comparisons. Include min-p, penalties, and seed where applicable. Do not describe different samplers as equivalent workloads.

8. **Correct the short-prompt and profile-context labels.**

   Status: confirmed discrepancies against saved logs.

   Locations: main.tex Table 2 caption, Table 6 caption, Table 7 caption, and Table 9 caption; [scripts/run_mlx.sh](scripts/run_mlx.sh), lines 6–9.

   The saved short-run logs show 20 input tokens for ds4 and llama.cpp, whereas the throughput caption says 22. The MLX short run reports 60 prompt tokens and emits thinking content. Its script applies the model chat template to a plain user message, whereas the ds4/llama.cpp script uses an explicit prompt with an empty thinking block.

   The short-context stage logs cover positions 20–30, not a fixed 40-token context. The 8k stage logs cover positions 8590–8600.

   Also, the saved long-prompt logs report 1450 tokens for llama.cpp and 1449 for MLX at the first context point. Establish the reason rather than assuming identical tokenization.

   Suggested change: save exact prompt token IDs, specify template and thinking settings, report the actual decode-position window, and either rerun a common prompt or accurately label the differing workloads. Audit each prompt label independently; the throughput-log discrepancy does not establish which prompt was used for the separate correctness table.

9. **Account for the different MLX output-head precision.**

   Status: confirmed confounder.

   Locations: main.tex lines 547–555, 618–629, and 694–708.

   The cached MLX checkpoint has a 4-bit affine output head. Its tensor headers show packed weights plus BF16 scales and biases, totaling 715,161,600 bytes. The local GGUF Q8_0 output head occupies 1,350,860,800 bytes.

   The difference is 635,699,200 bytes per head evaluation. At 620 GB/s, a simple byte-count estimate makes that approximately 1.0 ms, which is relevant to a roughly 1 token/s difference. This is an illustration of the confounder's scale, not a measurement of its actual latency contribution.

   Consequently, MLX versus ds4 does not isolate the cost of affine versus Q4_K layer dequantization. Prompt handling, output-head precision, quantization provenance, and engine implementation also differ.

   Suggested change: report per-component precision and streamed bytes for each engine. Either match the head format for a controlled comparison or present MLX as a practical baseline without attributing its rate difference solely to block format. The [MLX checkpoint configuration](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/raw/main/config.json) identifies its affine 4-bit group-64 format; the exact head sizes above came from local tensor headers.

10. **Do not say MLX reaches the paper's 41 tokens/s ceiling.**

    Status: internal terminology inconsistency.

    Locations: main.tex lines 43–46, 103–107, and 627–629.

    Approximately 34.1 tokens/s is comparable with the optimized ds4 rates, but it is not 41 tokens/s. Nor does similar complete-model throughput prove that the engines have the same limiting kernel.

    Suggested wording: “MLX achieves comparable observed decode throughput with its affine 4-bit conversion.” Discuss kernel-level effective bandwidth separately from complete-model throughput and qualify comparisons using item 9.

11. **Make the MLX perplexity scoring window match the pinned reference.**

    Status: confirmed implementation discrepancy.

    Locations: main.tex lines 422–426; [scripts/mlx_ppl.py](scripts/mlx_ppl.py), lines 14–22; pinned llama.cpp tools/perplexity/perplexity.cpp, lines 615–629 and 110–127.

    With n_ctx = 2048, the MLX script selects tok_lp[half - 1:], scoring 1024 predictions. The pinned llama.cpp code starts at logit position 1024 and scores targets at positions 1025 through 2047: 1023 predictions.

    To match that reference implementation, use tok_lp[half:]. Then verify identical token IDs, special-token handling, chunk boundaries, and recurrent/cache initialization.

    The saved MLX log reports 40,960 scored tokens, consistent with the current 1024-per-chunk script. The comparable reference count is 40,920. The likely numerical effect is small, but the protocols are not currently identical.

12. **Strengthen correctness checks before asserting engine-independent quality.**

    Status: evidence limitation.

    Locations: main.tex lines 229–236 and 379–407; [scripts/logits_dump.cpp](scripts/logits_dump.cpp); [scripts/cmp.py](scripts/cmp.py).

    Two prompts at decode step 4 are useful spot checks. They do not establish numerical equivalence over long generation, long contexts, split-KV boundaries, or every switch combination. The GGUF perplexities were computed with llama.cpp, so they do not validate ds4's behavior over those sequences.

    Suggested improvements:

    - Compare a shared, forced sequence of token IDs so that both engines always receive the same history.
    - If using independent greedy generation, explicitly verify agreement at every preceding step.
    - Exercise more decode steps and context lengths, including boundaries where the attention workgroup policy changes.
    - Compare prefill and incremental execution at a shared final prefix.
    - Run a same-file, same-token-sequence perplexity comparison between engines.
    - Report argmax agreement and distribution-sensitive measures alongside mean/max absolute logit differences.
    - Reject wrong-sized or non-finite dumps explicitly in the comparison script, and give failed checks a nonzero exit status.

    Replace “logits match” with the measured errors and test scope. Clarify whether “identical across optimizations” means complete vectors, aggregate metrics, or just rounded table entries.

    The dump switch is zero-based: step 0 is prefill, and step 4 is after four generated tokens have been fed back. Calling this simply the “4th forward” is ambiguous.

13. **Use paired quality comparisons and avoid claiming equivalence from overlapping errors.**

    Status: statistical interpretation issue.

    Locations: main.tex lines 401–407 and Table 4.

    Overlap between separately reported standard errors does not establish equivalence. These models score the same corpus, so the uncertainty of paired loss differences is more informative than comparing marginal error bars. Token losses are also correlated; the interpretation of a token-level standard error should be stated.

    Suggested change: preserve losses for shared examples and report paired differences, with uncertainty estimated at an appropriate chunk or document level. State the scoring protocol and the meaning of each error bar.

    Replace “free on this metric” with a description of the observed perplexity values and their limited evaluation scope. Report what the stated 3–4% penalty is relative to. Keep the ten-question downstream result as a coarse check, not evidence of equivalent quality or a ranking.

14. **Separate fusion gains from other optimizations.**

    Status: confirmed attribution error and methodological improvement.

    Locations: main.tex lines 93–102, Table 7, and lines 529–534.

    The 3.9 tokens/s all-on versus all-off difference includes vectorized matvecs and K-split, not just fusion. “Fusion is worth 3.9 tokens/s on its own” is therefore not supported by that ablation.

    Report ablations in milliseconds per token as well as tokens/s. Throughput differences are nonlinear; summing losses in tokens/s and comparing that sum with an end-to-end gain is not a clean additive-cost analysis.

    The fast-matvec and K-split switches also affect overlapping paths. Record the kernel selected under each configuration and add targeted paired-switch experiments if an interaction claim is central.

    Describe the current table as leave-one-out effects around the optimized build. Such effects need not equal standalone gains from the original implementation.

15. **Qualify the proposed kernel bottlenecks.**

    Status: causal claims stronger than the experiments establish.

    Locations: main.tex lines 98–108, 284–290, 618–629, 674–679, and 705–708.

    K-split being neutral or slower on selected shapes does not by itself prove occupancy is irrelevant. Changing rows per simdgroup or load width can change several aspects of execution at once.

    Similarly, similar Q4_0 and Q4_K gate/up effective bandwidth is evidence about those kernel implementations, not proof that dequantization has no remaining cost anywhere in the model. Q4_K and Q4_0 also have asymmetric FFN-down implementations: only Q4_K has the described K-split path.

    Suggested wording: the observations are “consistent with” the proposed activation-reuse or dequantization explanation. For a stronger conclusion, add shape-matched microbenchmarks, generated-code evidence, or relevant counters while controlling other kernel choices.

    Describe the Q4_0/Q4_K experiment as two quantizations of the same source checkpoint. Their reconstructed weight values differ, so “identical weights” is imprecise.

16. **Keep measured GPU spans distinct from profiler wall-clock overhead.**

    Status: clarification needed; the numerical GPU totals are supported.

    Locations: main.tex lines 205–216 and Table 2 caption.

    Reaggregating the saved logs reproduced:

    | Profile | Sum of per-stage GPU spans |
    | --- | ---: |
    | All off | 36.4303 ms/token |
    | All on | 30.9185 ms/token |
    | All on, 8k | 33.3909 ms/token |

    These match the manuscript's table after rounding.

    The instrumented runs themselves report much lower wall-clock generation throughput, for example 8.73 tokens/s for the all-on short profile. Therefore the “few percent” discussion must clearly refer to the summed GPU spans compared with an uninstrumented run, not total profiler overhead.

    Even a small difference between aggregate GPU spans and uninstrumented latency does not prove every stage is perturbed equally. State which values are measured directly, which come from another run, and which are estimates. Align prompt windows and token counting before making close comparisons.

17. **Improve repeatability and uncertainty reporting for small effects.**

    Status: experimental-design suggestion.

    Locations: main.tex Table 7 caption and Table 8 caption.

    Two ablation runs and single context-sweep runs are weak support for differences of 0.1–0.3 tokens/s. The 4k/8k inversion is described as within a ±1 tokens/s spread, but a reader needs the repeated-run evidence or an explanation of where that spread came from.

    Suggested change: use enough repeated runs to characterize the small effects being interpreted, alternate or randomize configuration order, and report the individual measurements with a defined uncertainty summary. State warmup, page-cache treatment, generation length, and whether the GPU was idle.

    Apply this first to the headline comparisons and borderline effects; a full device sweep is an extension rather than a prerequisite for correcting the present paper.

18. **Retain the fast-math finding, but limit its scope and isolate its contribution.**

    Status: mechanism reproduced; portability and attribution need qualification.

    Locations: main.tex lines 109–112, 362–377, and 747–748; ds4.c lines 40666–40674.

    The reviewer reproduced the following with Apple clang 17.0.0 (clang-1700.4.4.1), -O3, and -std=c99:

    | Implementation | With -ffast-math | With -ffast-math -fno-finite-math-only |
    | --- | --- | --- |
    | isfinite(float value) | Branch to library function ___isfinitef | Inline bit classification |
    | Exponent-bit test on a float argument | Folded to return true | Inline classification |
    | Exponent-bit test on bits loaded through a pointer | Integer-load classification survives | Integer-load classification survives |

    This supports the reported local compiler behavior. It does not establish that every compiler/SDK behaves the same way or that the memory-based workaround is an unconditional future optimization guarantee. [Clang's manual](https://clang.llvm.org/docs/UsersManual.html#cmdoption-ffast-math) documents the non-finite assumptions enabled by fast math.

    Include -fno-finite-math-only as a measured comparison, or consider compiling numerically sensitive sampling code with suitable flags separately. Report compiler and SDK versions.

    The final sampler also changes vectorization, tail-mass approximation, candidate handling, and fallback behavior. Do not assign the entire 2.2-to-0.5 ms gain to replacing isfinite without an isolated comparison. State approximation error and validate the resulting sampling behavior separately from forward-logit correctness.

    A minimal source reproducer is:

    ~~~c
    #include <math.h>
    #include <stdint.h>
    #include <string.h>

    int via_isfinite(float v) {
        return isfinite(v);
    }

    int via_value_bits(float v) {
        uint32_t b;
        memcpy(&b, &v, sizeof(b));
        return (b & 0x7f800000u) != 0x7f800000u;
    }

    int via_memory_bits(const float *p) {
        uint32_t b;
        memcpy(&b, p, sizeof(b));
        return (b & 0x7f800000u) != 0x7f800000u;
    }
    ~~~

    Compile to assembly with:

    ~~~sh
    clang -O3 -std=c99 -ffast-math -S finite.c -o finite-fast.s
    clang -O3 -std=c99 -ffast-math -fno-finite-math-only -S finite.c -o finite-honor.s
    ~~~

    This is a compiler-output reproducer, not a reproduction of the full sampler performance measurement or the historical unit-test failure count.

19. **Repair the reproduction paths and make failures visible.**

    Status: confirmed path defects.

    Locations: main.tex lines 762–828; [scripts/run_all.sh](scripts/run_all.sh), lines 6, 11, and 46; [scripts/run_q40.sh](scripts/run_q40.sh), lines 4–23; [scripts/run_eval.sh](scripts/run_eval.sh), lines 4–7; [scripts/README.md](scripts/README.md).

    The default DS4_ROOT is the parent of the working directory. When invoked as instructed from misc/paper/scripts, this resolves to misc/paper, which contains neither the ds4 executable nor gguf. Moving the artifact to paper/scripts would still make that default resolve to paper rather than the repository root.

    Additional defects:

    - run_all.sh looks for $PWD/prompt_1439.txt after changing to the repository root.
    - run_q40.sh uses $OLDPWD/cmp.py after changing from gguf to the repository root, making OLDPWD point at gguf.
    - The appendix finishes its build commands in the llama.cpp checkout but then runs ds4-relative download commands.
    - It changes into gguf and later uses a relative “cd paper/scripts” without returning to the repository.
    - The documented paper/scripts location differs from the inspected misc/paper/scripts location.
    - The reference-logit commands depend on a PROMPT variable whose exact assignment is not included inline.

    Suggested change: derive the script directory from the script's own path, anchor all other paths to explicit roots, quote path variables, and make the documented working-directory transitions unambiguous. Fail clearly on missing binaries, models, prompts, logs, or failed subprocesses. Missing prerequisites should not quietly produce partial tables.

20. **Pin and package the exact artifact inputs.**

    Status: reproducibility improvement.

    Locations: main.tex lines 151–156 and 764–799.

    Pin the ds4 artifact commit, llama.cpp commit, mlx and mlx-lm versions, Python environment, compiler/SDK, model revisions, and importance-matrix revision. The local llama.cpp checkout did match 6d9c82ea2bb34e277c0664b8dd3434bfb4dcfb27, dated September 9, 2026.

    Replace the floating “pip3 install -U mlx-lm” instruction with the tested dependency versions. The older DeepSeek file must have an exact retrieval location or revision; describing it as the same recipe without the -0731 suffix is not enough to retrieve the measured bytes reliably.

    Package the raw logs, exact prompts, token counts, command lines, and analysis inputs. Generate tables and figures from a shared machine-readable result set where practical, so captions and values do not drift.

    Check the artifact from its documented starting directory after fixing paths. Verify that the published branch includes the promised files; remote branch availability was not established by this review.

21. **Reconcile the remaining numerical and labeling inconsistencies.**

    Status: confirmed arithmetic or wording issues; some need clarification rather than a new experiment.

    | Location | Issue | Suggested correction |
    | --- | --- | --- |
    | main.tex lines 73–74 | 2 ms is called one seventh of approximately 30 ms | Approximately one fifteenth |
    | main.tex lines 461–463 | 26 ms of layer weight streaming is described as 560–620 GB/s, then the head is added separately | 13.70 GB / 26 ms is approximately 527 GB/s; if 26 ms includes the head, adding it again double-counts it |
    | main.tex line 463 | 33.1 tokens/s is labeled 63% of the nominal bound | Using 800 GB/s and 15.05 GB/token gives approximately 62.3%, before timing corrections |
    | main.tex lines 87–88 and 309–312 | Every optimization is said to have a switch | Specify the six switchable optimizations; the attention prologue remains present in both columns |
    | main.tex abstract and Table 6 versus Table 7 | Baseline alternates between 29.3 and 29.2 tokens/s | Label separate measurements or use a consistent aggregate; avoid “exactly” as a general reproducibility promise |
    | main.tex Table 2 caption | Short profile labeled 40-token context | Report the observed positions 20–30 |
    | main.tex lines 494–495 | A 0.3 tokens/s sampled-rate loss is directly associated with a 0.5 ms sampler | 1/32.8 - 1/33.1 corresponds to approximately 0.276 ms; explain distinct runs/paths or measure both components in a paired run |
    | Appendix reference-logit description | Step 4 called the fourth forward | Define step 0 as prefill and step k as after feeding back k generated tokens |

    These corrections should be made after choosing the final timing convention so that percentages and latency conversions agree throughout.

22. **Distinguish measured negative results from rejected designs.**

    Status: presentation and evidentiary improvement.

    Location: main.tex lines 672–690.

    The convolution/recurrence fusion appears to have been rejected from a traffic estimate rather than implemented and timed. Label it a rejected design or analytical estimate instead of presenting all three items as measured failed optimizations.

    Recheck the stated 32-fold redundant q/k convolution work. For the described design, 1536 threadgroups each computing 256 q/k channels, divided by 16 distinct q/k heads × 256 channels, gives 96 repetitions per unique q/k channel. The factor 32 is the number of groups per value head and omits the sharing of q/k heads across three value heads.

    This is a count of logical repeated work under that proposed design, not a measurement of DRAM traffic. State the denominator and expected caching behavior. The conclusion that redundant work is a concern can remain, but its basis should be explicit.

    The attention-gate fusion paragraph appropriately admits that function-constant specialization was not confirmed as the cause. Keep that distinction between an observed slowdown and a proposed explanation.

23. **Narrow broad architectural and novelty claims.**

    Status: framing suggestion.

    Locations: main.tex lines 115–139, 694–728.

    Present the contribution as the integration of measurement, correctness checking, model support, and controlled optimization evidence for this hardware/model combination. GPU timing, CPU critical-path effects, and context-dependent attention partitioning are not unique to unified memory.

    Define what “model-specific graph” and “model-specific fusion” mean operationally. A model implemented in Python still has model-specific architecture; the relevant distinction may be hand-written Metal scheduling versus framework primitives. State which relevant fusions the pinned MLX baseline uses before drawing architectural conclusions.

    The [Flash-Decoding description](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) explicitly concerns decode attention and partitions work along the KV sequence. Distinguish that from FlashAttention's original attention IO/training emphasis in the related-work paragraph.

    Similarly, saying CUDA kernels cannot be used directly on Metal is clearer than saying the chunked and recurrent mathematical forms “do not transfer.” Preserve clear attribution for ports and adaptations from llama.cpp/ggml and distinguish them from new implementation work.

24. **Qualify the speculative-decoding extension.**

    Status: scope and wording improvement.

    Locations: main.tex lines 708–710 and 734–737.

    The model does include an MTP layer; the local GGUF also contains its auxiliary tensors. However, verifying multiple candidate positions in one base-model pass does not guarantee two accepted output tokens per pass or a twofold speedup.

    Suggested change: describe the potential to amortize base-model weight reads across verified positions, with actual benefit depending on acceptance, drafting overhead, and the recurrent-state verification implementation. Keep this as future work unless measured.

25. **Improve the abstract and introduction.**

    Status: editorial suggestion.

    Locations: main.tex lines 24–50 and 91–129.

    Shorten the abstract, for example toward approximately 180–220 words if the venue permits, and retain the problem, measurement method, corrected headline result, main controlled finding, and evaluation scope. It currently asks the reader to track many rates, percentages, formats, and mechanisms before their definitions.

    Merge the overlapping “four findings” and “our contributions” lists. Introduce fewer kernel names before the model/background section.

    Prefer “a widely used runner” to an unsupported usage-ranking claim. Describe the demonstrated improvement precisely and leave uncertain bottleneck explanations out of the abstract until they have been established in the body.

26. **Make the measurement model easier to inspect.**

    Status: presentation suggestion.

    Add one diagram showing prefill completion, first-token sampling/emission, subsequent forward calls, CPU gaps, and GPU command-buffer spans. Mark which intervals each engine's existing throughput counter covers.

    A latency breakdown can then distinguish measured stage durations from the estimated weight-streaming lower bound. Avoid a visual that presents overlapping memory and compute activity as independently measured additive intervals.

    Define dispatch count, command-buffer count, CPU encode time, GPU span, and generation wall time separately. The current mixture of these quantities makes the claimed decomposition harder to audit than the underlying data warrants.

27. **Simplify the PDF figures and tables.**

    Status: visual review suggestion.

    All 16 pages were rendered and inspected. No unresolved “??” references or obvious clipping were found. The main presentation problems are density, redundant table/figure pairs, and floats separating related prose.

    Figure 2 uses a truncated horizontal bar axis, which visually enlarges relatively small rate differences. Prefer a dot plot of rates or a latency-delta plot with clearly defined uncertainty. If retaining bars, use an honest baseline or make the truncation conspicuous.

    Tables 2, 7, and 8 largely duplicate their corresponding figures. Keep the representation that best communicates each result in the main text and move detailed numerical counterparts to the appendix if space is tight.

    Table 6 is dense and mixes file-specific baselines with a ds4-versus-MLX comparison using different files. Separate or simplify it, and make the head format, timing definition, prompt, and sampler settings easy to find. Keep plot fonts legible at the final print size.

28. **Move the small downstream check and improve source specificity.**

    Status: editorial and reproducibility suggestions.

    The ten-question downstream check is appropriately caveated but weak evidence of comparative quality, particularly with several budget-limited outputs. Move it to the appendix unless expanded. Explain whether “correct” and “cut off” can overlap and how answers are extracted from truncated outputs.

    Replace the broad Qwen organization-page citation with the exact model card and configuration revision supporting the architecture. Include direct sources for the sampling defaults and toolchain behavior. Prefer fixed versions or revisions for implementation-dependent statements.

    The reviewed official Gated DeltaNet and AWQ proceedings entries agreed with the listed author sets; no author-list correction was established. The M5 discussion is plausible as a limitation/extension and can cite Apple's terminology for GPU Neural Accelerators if retained.

**Verified evidence retained from the review.**

The local Q4_K GGUF tensor metadata showed:

| Component | Bytes |
| --- | ---: |
| Base decoder layers, blocks 0–63 | 13,707,749,376 |
| Q8_0 output head | 1,350,860,800 |
| Output norm | 20,480 |
| Base decoder layers + head + output norm | 15,058,630,656 |
| Auxiliary MTP block | 238,983,168 |
| Full token-embedding table | 715,161,600 |

These numbers are tensor storage sizes, not measured total memory traffic. A normal embedding lookup reads a row rather than the full embedding table. Small non-matrix tensors explain why detailed storage accounting need not exactly match a rounded dominant-matrix estimate.

The saved perplexity values were:

| File | Saved result |
| --- | --- |
| Pure Q4_K | 5.8032 ± 0.06770 |
| UD-Q4_K_M | 5.8471 ± 0.06859 |
| Q8_0 | 5.8610 ± 0.06899 |
| Pure Q4_0 | 6.0291 ± 0.07223 |
| MLX affine 4-bit | 5.9820 over 40,960 scored tokens |

These agree with the manuscript's rounded entries. Items 11–13 concern protocol comparability and interpretation, not whether those logged values were transcribed correctly.

The saved ablation rates included all-off runs of 29.20 and 29.20 tokens/s, and all-on runs of 33.06 and 33.13 tokens/s. These support the rounded table entries under the existing timing definition.

The logs inspected were in:

~~~text
/private/tmp/claude-501/-Users-choco-github-ds4/da51aa41-25bf-4e5d-aea2-84a62e41db12/scratchpad/exp/logs/
~~~

Relevant files included stage_after.log, stage_alloff.log, stage_after_8k.log, llama_sampled_Q4.log, llama_sampled_Q8.log, llama_sampled_UD.log, llama_greedy_Q4.log, mlx_short.log, mlx_ctx_1k.log, mlx_ppl.log, ppl_Q4.log, ppl_Q8.log, ppl_UD.log, ppl_Q40.log, and abl_* logs. This temporary location is evidence provenance, not a suitable permanent artifact location.

The inspected MLX snapshot was:

~~~text
/Users/choco/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff/
~~~

Its output-head metadata was read from model-00003-of-00003.safetensors without running inference. The weight tensor had U32 shape [248320, 640], and the scale and bias tensors each had BF16 shape [248320, 80].

Primary references used for the factual checks include the [Qwen model card](https://huggingface.co/Qwen/Qwen3.8-27B), its [configuration](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json), its [generation configuration](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/generation_config.json), the [MLX checkpoint configuration](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/raw/main/config.json), and the [Clang manual](https://clang.llvm.org/docs/UsersManual.html#cmdoption-ffast-math). Implementation comparisons also used the locally pinned llama.cpp checkout and [mlx-lm v0.31.3 generation code](https://raw.githubusercontent.com/ml-explore/mlx-lm/v0.31.3/mlx_lm/generate.py).

**Suggested revision order.**

- [ ] Normalize throughput timing and first-token counting, then rerun the headline comparisons with matched prompts and sampling settings.
- [ ] Correct architecture, prompt/context labels, sampling defaults, roofline arithmetic, and derived percentages.
- [ ] Match the perplexity scoring protocol and add same-file cross-engine correctness evidence.
- [ ] Revise format, fusion, occupancy, and quality claims to match the controls and uncertainty actually available.
- [ ] Repair artifact paths, pin dependencies and model revisions, and package raw logs and exact commands.
- [ ] Shorten and reorganize the introduction, improve the timing diagram and ablation plot, and move secondary material to the appendix.

