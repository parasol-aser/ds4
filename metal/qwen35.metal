// Qwen3.5 family kernels (Qwen3.8-27B): Gated DeltaNet linear attention and
// the gated GQA attention prologue.  The delta-rule state update mirrors the
// GLM-5.3 KDA kernels; the differences are a scalar per-head decay, a fused
// q/k/v projection sharing one short convolution, 16 q/k heads serving 48
// value heads, and a SiLU-gated output norm.

struct qwen35_gdn_args {
    uint n_v_heads;      // value heads, one recurrent state each
    uint n_qk_heads;     // q/k heads; value head h reads q/k head h % QK
    uint n_tokens;
    uint conv_channels;  // 2 * qk_dim + v_dim, the fused q|k|v row width
    float eps;
};

static inline float qwen35_silu(float x) {
    return x / (1.0f + exp(-x));
}

static inline float qwen35_softplus(float x) {
    return x > 20.0f ? x : log(1.0f + exp(x));
}

/*
 * Causal depthwise conv (kernel 4) over the fused q|k|v row, then SiLU.
 * One thread owns one channel and walks the tokens in order, so the same
 * kernel serves decode (n_tokens = 1) and prefill.  The three-column
 * history holds raw projections, matching the reference conv state.
 * Weights are laid out channel-major with the four taps adjacent.
 */
kernel void kernel_qwen35_gdn_conv(
        constant qwen35_gdn_args &args,
        device float             *qkv,
        device const float       *conv_w,
        device float             *conv_state,
        uint c [[thread_position_in_grid]]) {
    const uint C = args.conv_channels;
    if (c >= C) return;
    const float4 w = *((device const float4 *)(conv_w + (ulong)c * 4u));
    float h0 = conv_state[c];
    float h1 = conv_state[C + c];
    float h2 = conv_state[2u * C + c];
    for (uint t = 0; t < args.n_tokens; t++) {
        const ulong index = (ulong)t * C + c;
        const float x = qkv[index];
        const float y = w.x * h0 + w.y * h1 + w.z * h2 + w.w * x;
        qkv[index] = qwen35_silu(y);
        h0 = h1;
        h1 = h2;
        h2 = x;
    }
    conv_state[c] = h0;
    conv_state[C + c] = h1;
    conv_state[2u * C + c] = h2;
}

/*
 * Delta-rule recurrence.  One simdgroup owns one value row of one head's
 * 128x128 state, every lane holds four key columns, and the token loop is
 * sequential.  q and k are L2-normalized here (each simdgroup covers a full
 * head, so the norm is one simd_sum) and q carries the 1/sqrt(128) scale.
 * decay = exp(A * softplus(alpha + dt_bias)) with A = -exp(A_log) already
 * folded into the GGUF ssm_a tensor; beta = sigmoid(b).
 */
kernel void kernel_qwen35_gdn_recurrence(
        constant qwen35_gdn_args &args,
        device const float       *qkv,
        device const float       *alpha,
        device const float       *beta,
        device const float       *ssm_a,
        device const float       *dt_bias,
        device float             *state,
        device float             *out,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    const uint head = tgpig.x;
    const uint value = tgpig.y * 4u + sg;
    if (head >= args.n_v_heads || value >= D) return;
    /* ggml tiles the q/k heads over the value heads (ggml_repeat), which is
     * the layout this GGUF was produced for. */
    const uint qk_head = head % args.n_qk_heads;
    const uint qk_dim = args.n_qk_heads * D;
    const uint v_dim = args.n_v_heads * D;
    const uint C = args.conv_channels;
    const uint k0 = lane * 4u;
    const float q_scale = 0x1.6a09e6p-4f;  // 1/sqrt(128)
    const float a = ssm_a[head];
    const float dt = dt_bias[head];

    device float4 *state_ptr = (device float4 *)(
        state + ((ulong)head * D + value) * D + k0);
    float4 h = *state_ptr;

    for (uint t = 0; t < args.n_tokens; t++) {
        const ulong row = (ulong)t * C;
        float4 q4 = *((device const float4 *)(qkv + row + qk_head * D + k0));
        float4 k4 = *((device const float4 *)(qkv + row + qk_dim + qk_head * D + k0));
        const float q_norm = sqrt(simd_sum(dot(q4, q4)));
        const float k_norm = sqrt(simd_sum(dot(k4, k4)));
        q4 *= q_scale / max(q_norm, args.eps);
        k4 /= max(k_norm, args.eps);
        const ulong gate_index = (ulong)t * args.n_v_heads + head;
        const float decay = exp(a * qwen35_softplus(alpha[gate_index] + dt));
        const float b = 1.0f / (1.0f + exp(-beta[gate_index]));

        h *= decay;
        const float hk = simd_sum(dot(h, k4));
        const float v = qkv[row + 2u * qk_dim + head * D + value];
        h = fma(k4, float4((v - hk) * b), h);
        const float o = simd_sum(dot(h, q4));
        if (lane == 0u) out[(ulong)t * v_dim + head * D + value] = o;
    }
    *state_ptr = h;
}

/* Per-head RMSNorm of the recurrence output, gated by SiLU(z). */
kernel void kernel_qwen35_gdn_output(
        constant qwen35_gdn_args &args,
        device float             *out,
        device const float       *z,
        device const float       *norm_w,
        threadgroup float        *partial [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    const uint token = tgpig.x;
    const uint head = tgpig.y;
    if (token >= args.n_tokens || head >= args.n_v_heads) return;
    const ulong index = ((ulong)token * args.n_v_heads + head) * D + tid;
    const float raw = out[index];
    const float sumsq = simd_sum(raw * raw);
    if (lane == 0u) partial[sg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = lane < 4u ? partial[lane] : 0.0f;
    total = simd_sum(total);
    const float scale = rsqrt(total / (float)D + args.eps);
    out[index] = raw * scale * norm_w[tid] * qwen35_silu(z[index]);
}

struct qwen35_attn_args {
    uint n_tokens;
    uint n_head;
    uint src_row_stride;   // floats between tokens in src
    uint src_head_stride;  // floats between heads in src
    uint dst_row_stride;   // elements between rows in dst
    uint dst_row0;         // first destination row (cache slot of token 0)
    uint pos0;             // RoPE position of token 0
    uint n_rot;            // rotated prefix width (64)
    float freq_base;
    float eps;
};

/*
 * Attention head prologue for one q or k head: per-head RMSNorm with weight,
 * then NEOX-style RoPE on the leading n_rot dims (pairs i, i + n_rot/2).
 * src and dst already point at the head; the threadgroup is the 256 dims.
 */
template<typename T>
static inline void qwen35_head_norm_rope(
        device const float *src,
        device const float *norm_w,
        device T           *dst,
        threadgroup float  *shared,
        float               pos,
        uint                n_rot,
        float               freq_base,
        float               eps,
        ushort              tid,
        ushort              lane,
        ushort              sg) {
    constexpr uint D = 256u;
    threadgroup float *vals = shared;
    threadgroup float *partial = shared + D;
    const float x = src[tid];
    const float sumsq = simd_sum(x * x);
    if (lane == 0u) partial[sg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = lane < 8u ? partial[lane] : 0.0f;
    total = simd_sum(total);
    const float scale = rsqrt(total / (float)D + eps);
    vals[tid] = x * scale * norm_w[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint half_rot = n_rot / 2u;
    if (tid < half_rot) {
        const float theta = pos * pow(freq_base, -(float)tid / (float)half_rot);
        const float c = cos(theta);
        const float s = sin(theta);
        const float y0 = vals[tid];
        const float y1 = vals[tid + half_rot];
        vals[tid] = y0 * c - y1 * s;
        vals[tid + half_rot] = y0 * s + y1 * c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dst[tid] = (T)vals[tid];
}

struct qwen35_attn_prologue_args {
    uint n_tokens;
    uint n_head;
    uint n_head_kv;
    uint q_row_stride;     // floats between tokens in the [q | gate] source
    uint q_head_stride;    // floats between q heads (512: q then gate)
    uint k_row_stride;     // floats between tokens in the k source
    uint v_row_stride;     // floats between tokens in the v source
    uint cache_row_stride; // elements between cache rows (n_head_kv * 256)
    uint dst_row0;         // cache slot of token 0
    uint pos0;             // RoPE position of token 0
    uint n_rot;
    float freq_base;
    float eps;
};

/*
 * Whole attention prologue in one grid: threadgroup.y < n_head prepares a
 * query head into the f32 scratch, the next n_head_kv prepare a key head
 * into the f16 K cache, the last n_head_kv copy a value head into the V
 * cache.  Decode feeds the k and v sources from the multi-weight
 * projection's tail; prefill from their own batch buffers.
 */
kernel void kernel_qwen35_attn_prologue(
        constant qwen35_attn_prologue_args &args,
        device const float                 *q_src,
        device const float                 *k_src,
        device const float                 *v_src,
        device const float                 *q_norm,
        device const float                 *k_norm,
        device float                       *q_dst,
        device half                        *k_cache,
        device half                        *v_cache,
        threadgroup float                  *shared [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 256u;
    const uint token = tgpig.x;
    const uint slot = tgpig.y;
    if (token >= args.n_tokens) return;
    const ulong cache_row = (ulong)(args.dst_row0 + token) * args.cache_row_stride;
    const float pos = (float)(args.pos0 + token);
    if (slot < args.n_head) {
        qwen35_head_norm_rope<float>(
            q_src + (ulong)token * args.q_row_stride + (ulong)slot * args.q_head_stride,
            q_norm,
            q_dst + ((ulong)token * args.n_head + slot) * D,
            shared, pos, args.n_rot, args.freq_base, args.eps, tid, lane, sg);
    } else if (slot < args.n_head + args.n_head_kv) {
        const uint head = slot - args.n_head;
        qwen35_head_norm_rope<half>(
            k_src + (ulong)token * args.k_row_stride + (ulong)head * D,
            k_norm,
            k_cache + cache_row + (ulong)head * D,
            shared, pos, args.n_rot, args.freq_base, args.eps, tid, lane, sg);
    } else if (slot < args.n_head + 2u * args.n_head_kv) {
        const uint head = slot - args.n_head - args.n_head_kv;
        v_cache[cache_row + (ulong)head * D + tid] =
            (half)v_src[(ulong)token * args.v_row_stride + (ulong)head * D + tid];
    }
}

/* Gated attention output: heads *= sigmoid(gate), gate taken from the
 * interleaved [q | gate] projection row. */
kernel void kernel_qwen35_attn_gate(
        constant qwen35_attn_args &args,
        device float              *heads,
        device const float        *qg,
        uint gid [[thread_position_in_grid]]) {
    constexpr uint D = 256u;
    const ulong total = (ulong)args.n_tokens * args.n_head * D;
    if (gid >= total) return;
    const uint token = gid / (args.n_head * D);
    const uint rem = gid - token * (args.n_head * D);
    const uint head = rem / D;
    const uint d = rem - head * D;
    const float gate = qg[(ulong)token * args.src_row_stride +
                          (ulong)head * args.src_head_stride + D + d];
    heads[gid] *= 1.0f / (1.0f + exp(-gate));
}
