#include "operator_base.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <arm_neon.h>
#include "rknn_api.h"
#include "rknn_matmul_api.h"

void matmul_init(OperatorTest* op) {
    op->input_dims[0] = 1;
    op->input_dims[1] = 64;
    op->input_dims[2] = 1;
    op->input_dims[3] = 128;
    op->input_size = op->input_dims[0] * op->input_dims[1] * op->input_dims[2] * op->input_dims[3] * sizeof(float);

    int K = 128;
    int N = 64;
    op->output_dims[0] = 1;
    op->output_dims[1] = 64;
    op->output_dims[2] = 1;
    op->output_dims[3] = N;
    op->output_size = op->output_dims[0] * op->output_dims[1] * op->output_dims[2] * op->output_dims[3] * sizeof(float);

    int weight_size = K * N * sizeof(float);
    op->weight_data = (float*)malloc(weight_size);
    op->input_data = (float*)malloc(op->input_size);
    op->cpu_output = (float*)malloc(op->output_size);
    op->npu_output = (float*)malloc(op->output_size);

    unsigned int seed = 42;
    op->seed = seed;
    srand(seed);

    for (int i = 0; i < op->input_size / sizeof(float); i++) {
        op->input_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    for (int i = 0; i < weight_size / sizeof(float); i++) {
        op->weight_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

void matmul_run_cpu(OperatorTest* op) {
    int M = op->input_dims[1];
    int K = op->input_dims[3];
    int N = op->output_dims[3];

    memset(op->cpu_output, 0, op->output_size);

    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            float a_val = op->input_data[m*K + k];
            if (a_val == 0.0f) continue;
            for (int n = 0; n < N; n++) {
                op->cpu_output[m*N + n] += a_val * op->weight_data[k*N + n];
            }
        }
    }
}

static void float32_to_float16(const float* src, uint16_t* dst, int count) {
    for (int i = 0; i < count; i += 8) {
        float32x4_t f32_0 = vld1q_f32(src + i);
        float32x4_t f32_1 = vld1q_f32(src + i + 4);
        float16x4_t f16_0 = vcvt_f16_f32(f32_0);
        float16x4_t f16_1 = vcvt_f16_f32(f32_1);
        vst1_u16(dst + i, vreinterpret_u16_f16(f16_0));
        vst1_u16(dst + i + 4, vreinterpret_u16_f16(f16_1));
    }
}

int rknn_matmul_forward(rknn_context ctx,
                        const float* A,
                        const float* B,
                        float* C,
                        int M, int K, int N) {
    rknn_matmul_ctx matmul_ctx;
    rknn_matmul_info info = {0};
    rknn_matmul_io_attr io_attr = {0};

    info.M = M;
    info.K = K;
    info.N = N;
    info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    info.B_layout = RKNN_MM_LAYOUT_NORM;
    info.AC_layout = RKNN_MM_LAYOUT_NORM;

    int ret = rknn_matmul_create(&matmul_ctx, &info, &io_attr);
    if (ret != 0) {
        printf("❌ rknn_matmul_create 失败，错误码：%d\n", ret);
        return ret;
    }

    int A_elem = M * K;
    int B_elem = K * N;
    int C_elem = M * N;

    size_t A_size_f16 = A_elem * sizeof(uint16_t);
    size_t B_size_f16 = B_elem * sizeof(uint16_t);
    size_t C_size_f32 = C_elem * sizeof(float);

    rknn_tensor_mem* A_mem = rknn_create_mem(ctx, A_size_f16);
    rknn_tensor_mem* B_mem = rknn_create_mem(ctx, B_size_f16);
    rknn_tensor_mem* C_mem = rknn_create_mem(ctx, C_size_f32);
    if (!A_mem || !B_mem || !C_mem) {
        printf("❌ 内存分配失败\n");
        if (A_mem) rknn_destroy_mem(ctx, A_mem);
        if (B_mem) rknn_destroy_mem(ctx, B_mem);
        if (C_mem) rknn_destroy_mem(ctx, C_mem);
        rknn_matmul_destroy(matmul_ctx);
        return -1;
    }

    uint16_t* A_f16 = (uint16_t*)malloc(A_size_f16);
    uint16_t* B_f16 = (uint16_t*)malloc(B_size_f16);
    if (!A_f16 || !B_f16) {
        printf("❌ 临时内存分配失败\n");
        free(A_f16); free(B_f16);
        rknn_destroy_mem(ctx, A_mem);
        rknn_destroy_mem(ctx, B_mem);
        rknn_destroy_mem(ctx, C_mem);
        rknn_matmul_destroy(matmul_ctx);
        return -1;
    }

    float32_to_float16(A, A_f16, A_elem);
    float32_to_float16(B, B_f16, B_elem);

    memcpy(A_mem->virt_addr, A_f16, A_size_f16);
    memcpy(B_mem->virt_addr, B_f16, B_size_f16);
    free(A_f16); free(B_f16);

    ret = rknn_matmul_set_io_mem(matmul_ctx, A_mem, &io_attr.A);
    if (ret != 0) { printf("❌ 绑定 A 失败：%d\n", ret); goto cleanup; }
    ret = rknn_matmul_set_io_mem(matmul_ctx, B_mem, &io_attr.B);
    if (ret != 0) { printf("❌ 绑定 B 失败：%d\n", ret); goto cleanup; }
    ret = rknn_matmul_set_io_mem(matmul_ctx, C_mem, &io_attr.C);
    if (ret != 0) { printf("❌ 绑定 C 失败：%d\n", ret); goto cleanup; }

    ret = rknn_matmul_run(matmul_ctx);
    if (ret != 0) { printf("❌ 执行失败：%d\n", ret); goto cleanup; }

    ret = rknn_mem_sync(ctx, C_mem, RKNN_MEMORY_SYNC_FROM_DEVICE);
    if (ret != 0) { printf("❌ 同步失败：%d\n", ret); goto cleanup; }

    memcpy(C, C_mem->virt_addr, C_size_f32);

cleanup:
    if (A_mem) rknn_destroy_mem(ctx, A_mem);
    if (B_mem) rknn_destroy_mem(ctx, B_mem);
    if (C_mem) rknn_destroy_mem(ctx, C_mem);
    rknn_matmul_destroy(matmul_ctx);
    return ret;
}

int matmul_run_npu(OperatorTest* op, rknn_context ctx) {
    int M = op->input_dims[1];
    int K = op->input_dims[3];
    int N = op->output_dims[3];
    int ret = rknn_matmul_forward(ctx, op->input_data, op->weight_data, op->npu_output, M, K, N);
    if (ret != 0) {
        printf("❌ MatMul：NPU执行失败，错误码：%d\n", ret);
    }
    return ret;
}

void register_matmul_operator() {
    OperatorInterface iface = {
        .init = matmul_init,
        .run_cpu = matmul_run_cpu,
        .run_npu = matmul_run_npu
    };
    register_operator("MatMul", iface);
}
