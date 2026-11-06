// v3 (= v1 + K loop unrolled by two tiles with constant stage index, pointer-increment global loads): A staged transposed as plain half (two rows packed per 32-bit store, XOR-swizzled pairs), one LDS.128 of A
// (8 rows) and one LDS.128 of B (8 columns) per k feed 32 HFMA2 (the A broadcast is the HFMA2 operand selector).
// Thread owns rows ty*8 + {0..7} and columns tx*8 + {0..7}. Same per-element arithmetic as the base kernel.
template <bool OUT_F32, int BM, int STAGES = 2, int MINB = 128 / BM, int GROUP_M = 0>
__global__ void __launch_bounds__(BM * 2, MINB)
gemm_kernel_v3
(
    const half* __restrict__ A,
    const half* __restrict__ B,
    void* __restrict__ C,
    const int M,
    const int N,
    const int K,
    const int ldc,
    const int64_t a_bstride,
    const int64_t b_bstride,
    const int64_t c_bstride
)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 600
    __shared__ __align__(16) half As[STAGES][BK][BM];
    __shared__ __align__(16) half Bs[STAGES][BK][BN];

    A += blockIdx.z * a_bstride;
    B += blockIdx.z * b_bstride;

    constexpr int THREADS = BM * 2;
    constexpr int LB = BK * BN / 8 / THREADS;      // 16-byte B loads per thread per K tile
    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    if constexpr (GROUP_M > 0)
    {
        // Grouped raster: consecutive blocks walk GROUP_M row stripes down one column stripe, so a wave shares
        // B stripes across GROUP_M row stripes in L2 instead of streaming all of B once per row stripe
        const int grid_n = gridDim.x, grid_m = gridDim.y;
        const int bid = blockIdx.y * grid_n + blockIdx.x;
        const int group = bid / (GROUP_M * grid_n);
        const int first_m = group * GROUP_M;
        const int gm = min(GROUP_M, grid_m - first_m);
        const int in_group = bid - group * GROUP_M * grid_n;
        m0 = (first_m + in_group % gm) * BM;
        n0 = (in_group / gm) * BN;
    }

    // A: row pair p = tid / 4 (rows 2p, 2p + 1), k chunk c = tid % 4 (k = 8c .. 8c + 7). Stored at pair p ^ 8c of
    // row k so the eight k stores of a warp hit 32 distinct banks
    const int a_p = tid >> 2, a_c = tid & 3;
    const bool a_ok0 = m0 + 2 * a_p < M, a_ok1 = m0 + 2 * a_p + 1 < M;
    const half* __restrict__ a_ptr0 = A + (size_t) (a_ok0 ? m0 + 2 * a_p : 0) * K + a_c * 8;
    const half* __restrict__ a_ptr1 = A + (size_t) (a_ok1 ? m0 + 2 * a_p + 1 : 0) * K + a_c * 8;
    const int a_sp = (a_p ^ (a_c * 8)) * 2;
    const int b_k = tid >> 4, b_n = (tid & 15) * 8;
    const half* b_ptr = B + (size_t) b_k * N + n0 + b_n;
    const size_t b_step = (size_t) BK * N;

    half2 acc[8][4];
    float accf[8][8];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        #pragma unroll
        for (int j = 0; j < 8; ++j) accf[i][j] = 0.0f;
    }

    uint4 ra[2], rb[LB];
    // Loads the next K tile and advances the pointers
    auto gload = [&] ()
    {
        ra[0] = a_ok0 ? *reinterpret_cast<const uint4*>(a_ptr0) : make_uint4(0, 0, 0, 0);
        ra[1] = a_ok1 ? *reinterpret_cast<const uint4*>(a_ptr1) : make_uint4(0, 0, 0, 0);
        #pragma unroll
        for (int l = 0; l < LB; ++l)
            rb[l] = *reinterpret_cast<const uint4*>(b_ptr + (size_t) l * (THREADS / 16) * N);
        a_ptr0 += BK; a_ptr1 += BK; b_ptr += b_step;
    };
    auto sstore = [&] (int buf)
    {
        const uint32_t* r0 = reinterpret_cast<const uint32_t*>(&ra[0]);
        const uint32_t* r1 = reinterpret_cast<const uint32_t*>(&ra[1]);
        #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            const uint32_t w = __byte_perm(r0[i / 2], r1[i / 2], (i & 1) ? 0x7632 : 0x5410);
            *reinterpret_cast<uint32_t*>(&As[buf][a_c * 8 + i][a_sp]) = w;
        }
        #pragma unroll
        for (int l = 0; l < LB; ++l)
            *reinterpret_cast<uint4*>(&Bs[buf][l * (THREADS / 16) + b_k][b_n]) = rb[l];
    };

    const int KT = K / BK;
    gload();
    sstore(0);
    __syncthreads();

    // One K tile from stage buf (a constant after inlining); more: another tile follows
    auto tile = [&] (const int buf, const bool more)
    {
        if (more) gload();

        #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = __float2half2_rn(0.0f);
        }

        #pragma unroll
        for (int k = 0; k < BK; ++k)
        {
            const uint4 a = *reinterpret_cast<const uint4*>(&As[buf][k][((ty * 4) ^ ((k >> 3) * 8)) * 2]);
            const uint4 b = *reinterpret_cast<const uint4*>(&Bs[buf][k][tx * 8]);
            const half2* av = reinterpret_cast<const half2*>(&a);
            const half2* bv = reinterpret_cast<const half2*>(&b);
            #pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                const half2 ai = (i & 1) ? __high2half2(av[i / 2]) : __low2half2(av[i / 2]);
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = __hfma2(ai, bv[j], acc[i][j]);
            }
        }

        #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                const float2 f = __half22float2(acc[i][j]);
                accf[i][2 * j] += f.x;
                accf[i][2 * j + 1] += f.y;
            }
        }

        if constexpr (STAGES == 1) __syncthreads();
        if (more) sstore(STAGES == 2 ? buf ^ 1 : 0);
        __syncthreads();
    };

    int kt = 0;
    for (; kt + 2 <= KT; kt += 2)
    {
        tile(0, true);
        tile(STAGES == 2 ? 1 : 0, kt + 2 < KT);
    }
    if (kt < KT) tile(0, false);

    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        const int m = m0 + ty * 8 + i;
        if (m >= M) continue;
        if constexpr (OUT_F32)
        {
            float* c = reinterpret_cast<float*>(C) + blockIdx.z * c_bstride + (size_t) m * ldc + n0 + tx * 8;
            *reinterpret_cast<float4*>(c) = make_float4(accf[i][0], accf[i][1], accf[i][2], accf[i][3]);
            *reinterpret_cast<float4*>(c + 4) = make_float4(accf[i][4], accf[i][5], accf[i][6], accf[i][7]);
        }
        else
        {
            half* c = reinterpret_cast<half*>(C) + blockIdx.z * c_bstride + (size_t) m * ldc + n0 + tx * 8;
            half2 h[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) h[j] = __floats2half2_rn(accf[i][2 * j], accf[i][2 * j + 1]);
            *reinterpret_cast<uint2*>(c) = *reinterpret_cast<const uint2*>(&h[0]);
            *reinterpret_cast<uint2*>(c + 4) = *reinterpret_cast<const uint2*>(&h[2]);
        }
    }
#endif
}
