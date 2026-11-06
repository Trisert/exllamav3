template <bool OUT_F32, int BM>
__global__ void __launch_bounds__(BM * 2, 128 / BM)
gemm_kernel
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
    __shared__ __align__(16) half2 As[2][BK][BM];
    __shared__ __align__(16) half Bs[2][BK][BN];

    A += blockIdx.z * a_bstride;
    B += blockIdx.z * b_bstride;

    constexpr int THREADS = BM * 2;
    constexpr int LB = BK * BN / 8 / THREADS;      // 16-byte B loads per thread per K tile
    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

    // Global load mapping, two 16-deep halves of the K tile: A row tid / 2, k (tid & 1) * 8 + {0, 16};
    // B rows tid / 16 + l * THREADS / 16, columns (tid & 15) * 8
    const int a_row = tid >> 1, a_k = (tid & 1) * 8;
    const bool a_ok = m0 + a_row < M;
    const half* a_ptr = A + (size_t) (a_ok ? m0 + a_row : 0) * K + a_k;
    const int b_k = tid >> 4, b_n = (tid & 15) * 8;
    const half* b_ptr = B + (size_t) b_k * N + n0 + b_n;

    half2 acc[8][4];
    float accf[8][8];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        #pragma unroll
        for (int j = 0; j < 8; ++j) accf[i][j] = 0.0f;
    }

    uint4 ra[2], rb[LB];
    auto gload = [&] (int kt)
    {
        #pragma unroll
        for (int l = 0; l < 2; ++l)
            ra[l] = a_ok ? *reinterpret_cast<const uint4*>(a_ptr + kt * BK + l * 16) : make_uint4(0, 0, 0, 0);
        #pragma unroll
        for (int l = 0; l < LB; ++l)
            rb[l] = *reinterpret_cast<const uint4*>(b_ptr + (size_t) (kt * BK + l * (THREADS / 16)) * N);
    };
    auto sstore = [&] (int buf)
    {
        #pragma unroll
        for (int l = 0; l < 2; ++l)
        {
            const half* ha = reinterpret_cast<const half*>(&ra[l]);
            #pragma unroll
            for (int i = 0; i < 8; ++i) As[buf][l * 16 + a_k + i][a_row] = __half2half2(ha[i]);
        }
        #pragma unroll
        for (int l = 0; l < LB; ++l)
            *reinterpret_cast<uint4*>(&Bs[buf][l * (THREADS / 16) + b_k][b_n]) = rb[l];
    };

    const int KT = K / BK;
    gload(0);
    sstore(0);
    __syncthreads();

    for (int kt = 0; kt < KT; ++kt)
    {
        const int buf = kt & 1;
        if (kt + 1 < KT) gload(kt + 1);

        // One fp16 partial per K tile (KSLICE = BK = 32 terms)
        #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = __float2half2_rn(0.0f);
        }

        #pragma unroll
        for (int k = 0; k < BK; ++k)
        {
            const uint4 a0 = *reinterpret_cast<const uint4*>(&As[buf][k][ty * 4]);
            const uint4 a1 = *reinterpret_cast<const uint4*>(&As[buf][k][BM / 2 + ty * 4]);
            const uint2 b0 = *reinterpret_cast<const uint2*>(&Bs[buf][k][tx * 4]);
            const uint2 b1 = *reinterpret_cast<const uint2*>(&Bs[buf][k][64 + tx * 4]);
            const half2* av0 = reinterpret_cast<const half2*>(&a0);
            const half2* av1 = reinterpret_cast<const half2*>(&a1);
            const half2* bv0 = reinterpret_cast<const half2*>(&b0);
            const half2* bv1 = reinterpret_cast<const half2*>(&b1);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                acc[i][0] = __hfma2(av0[i], bv0[0], acc[i][0]);
                acc[i][1] = __hfma2(av0[i], bv0[1], acc[i][1]);
                acc[i][2] = __hfma2(av0[i], bv1[0], acc[i][2]);
                acc[i][3] = __hfma2(av0[i], bv1[1], acc[i][3]);
                acc[4 + i][0] = __hfma2(av1[i], bv0[0], acc[4 + i][0]);
                acc[4 + i][1] = __hfma2(av1[i], bv0[1], acc[4 + i][1]);
                acc[4 + i][2] = __hfma2(av1[i], bv1[0], acc[4 + i][2]);
                acc[4 + i][3] = __hfma2(av1[i], bv1[1], acc[4 + i][3]);
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

        if (kt + 1 < KT) sstore(buf ^ 1);
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        const int m = m0 + (i < 4 ? ty * 4 + i : BM / 2 + ty * 4 + i - 4);
        if (m >= M) continue;
        if constexpr (OUT_F32)
        {
            float* c = reinterpret_cast<float*>(C) + blockIdx.z * c_bstride + (size_t) m * ldc + n0;
            *reinterpret_cast<float4*>(c + tx * 4) = make_float4(accf[i][0], accf[i][1], accf[i][2], accf[i][3]);
            *reinterpret_cast<float4*>(c + 64 + tx * 4) = make_float4(accf[i][4], accf[i][5], accf[i][6], accf[i][7]);
        }
        else
        {
            half* c = reinterpret_cast<half*>(C) + blockIdx.z * c_bstride + (size_t) m * ldc + n0;
            half2 h[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) h[j] = __floats2half2_rn(accf[i][2 * j], accf[i][2 * j + 1]);
            *reinterpret_cast<uint2*>(c + tx * 4) = *reinterpret_cast<const uint2*>(&h[0]);
            *reinterpret_cast<uint2*>(c + 64 + tx * 4) = *reinterpret_cast<const uint2*>(&h[2]);
        }
    }
#endif
}
