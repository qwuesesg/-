#include "operator_base.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <arm_neon.h>
#include "rknn_api.h"
#include "rknn_matmul_api.h"

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

static void float16_to_float32(const uint16_t* src, float* dst, int count) {
    for (int i = 0; i < count; i += 8) {
        uint16x4_t f16_0 = vld1_u16(src + i);
        uint16x4_t f16_1 = vld1_u16(src + i + 4);
        float32x4_t f32_0 = vcvt_f32_f16(vreinterpret_f16_u16(f16_0));
        float32x4_t f32_1 = vcvt_f32_f16(vreinterpret_f16_u16(f16_1));
        vst1q_f32(dst + i, f32_0);
        vst1q_f32(dst + i + 4, f32_1);
    }
}

void rope_init(OperatorTest* op) {
    op->input_dims[0] = 1; op->input_dims[1] = 128;
    op->input_dims[2] = 12; op->input_dims[3] = 64;
    op->input_size = 1 * 128 * 12 * 64 * sizeof(float);
    memcpy(op->output_dims, op->input_dims, sizeof(op->input_dims));
    op->output_size = op->input_size;
    op->input_data = (float*)malloc(op->input_size);
    op->weight_data = NULL;
    op->cpu_output = (float*)malloc(op->output_size);
    op->npu_output = (float*)malloc(op->output_size);
    unsigned int seed = (unsigned int)time(NULL);
    op->seed = seed; srand(seed);
    int elem = op->input_size / sizeof(float);
    for (int i = 0; i < elem; i++)
        op->input_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

static void generate_cos_sin(int seq_len, int head_dim, float* cos, float* sin) {
    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < head_dim / 2; i++) {
            float angle = pos / pow(10000.0, 2.0 * i / head_dim);
            cos[pos * head_dim + i] = cosf(angle);
            cos[pos * head_dim + head_dim/2 + i] = cosf(angle);
            sin[pos * head_dim + i] = sinf(angle);
            sin[pos * head_dim + head_dim/2 + i] = sinf(angle);
        }
    }
}

void rope_run_cpu(OperatorTest* op) {
    int batch = op->input_dims[0], seq_len = op->input_dims[1];
    int num_heads = op->input_dims[2], head_dim = op->input_dims[3];
    int M = batch * seq_len * num_heads;
    int K = head_dim;
    float* x = op->input_data; float* out = op->cpu_output;
    float* cos_base = (float*)malloc(seq_len * head_dim * sizeof(float));
    float* sin_base = (float*)malloc(seq_len * head_dim * sizeof(float));
    generate_cos_sin(seq_len, head_dim, cos_base, sin_base);
    for (int m = 0; m < M; m++) {
        int s = m / num_heads;
        int offset = m * K;
        for (int i = 0; i < K; i += 2) {
            float x1 = x[offset + i], x2 = x[offset + i + 1];
            float c = cos_base[s * K + i];
            float si = sin_base[s * K + i];
            out[offset + i] = x1 * c - x2 * si;
            out[offset + i + 1] = x1 * si + x2 * c;
        }
    }
    free(cos_base); free(sin_base);
}

int rope_run_npu(OperatorTest* op, rknn_context ctx) {
    static rknn_context rope_ctx = 0;
    if (rope_ctx == 0) {
        int ret = rknn_init(&rope_ctx, (void*)"./models/RoPE.rknn", 0, 0, NULL);
        if (ret != 0) { printf("❌ RoPE: load model failed, error %d\n", ret); return ret; }
    }

    rknn_input_output_num io_num;
    rknn_query(rope_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    rknn_tensor_attr input_attrs[3], output_attr;
    for (int i = 0; i < 3; i++) {
        memset(&input_attrs[i], 0, sizeof(rknn_tensor_attr));
        input_attrs[i].index = i;
        rknn_query(rope_ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
    }
    memset(&output_attr, 0, sizeof(output_attr)); output_attr.index = 0;
    rknn_query(rope_ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(rknn_tensor_attr));

    rknn_tensor_mem *xm = rknn_create_mem(rope_ctx, input_attrs[0].size);
    rknn_tensor_mem *cm = rknn_create_mem(rope_ctx, input_attrs[1].size);
    rknn_tensor_mem *sm = rknn_create_mem(rope_ctx, input_attrs[2].size);
    rknn_tensor_mem *om = rknn_create_mem(rope_ctx, output_attr.size);

    int seq_len = 128, num_heads = 12, head_dim = 64;
    int M = seq_len * num_heads, K = head_dim;

    float *cb = malloc(seq_len * K * sizeof(float));
    float *sb = malloc(seq_len * K * sizeof(float));
    generate_cos_sin(seq_len, head_dim, cb, sb);

    float *cf = malloc(M * K * sizeof(float));
    float *sf = malloc(M * K * sizeof(float));
    for (int s = 0; s < seq_len; s++)
        for (int h = 0; h < num_heads; h++) {
            memcpy(cf + (s * num_heads + h) * K, cb + s * K, K * sizeof(float));
            memcpy(sf + (s * num_heads + h) * K, sb + s * K, K * sizeof(float));
        }
    free(cb); free(sb);

    uint16_t *xf = malloc(input_attrs[0].size);
    uint16_t *cf16 = malloc(input_attrs[1].size);
    uint16_t *sf16 = malloc(input_attrs[2].size);
    float32_to_float16(op->input_data, xf, M * K);
    float32_to_float16(cf, cf16, M * K);
    float32_to_float16(sf, sf16, M * K);
    memcpy(xm->virt_addr, xf, input_attrs[0].size);
    memcpy(cm->virt_addr, cf16, input_attrs[1].size);
    memcpy(sm->virt_addr, sf16, input_attrs[2].size);
    free(xf); free(cf16); free(sf16); free(cf); free(sf);

    rknn_set_io_mem(rope_ctx, xm, &input_attrs[0]);
    rknn_set_io_mem(rope_ctx, cm, &input_attrs[1]);
    rknn_set_io_mem(rope_ctx, sm, &input_attrs[2]);
    rknn_set_io_mem(rope_ctx, om, &output_attr);
    rknn_run(rope_ctx, NULL);
    rknn_mem_sync(rope_ctx, om, RKNN_MEMORY_SYNC_FROM_DEVICE);

    int total_float32 = output_attr.size / sizeof(uint16_t);
    float* npu_raw = malloc(total_float32 * sizeof(float));
    uint16_t* out_half = (uint16_t*)om->virt_addr;
    float16_to_float32(out_half, npu_raw, total_float32);

    // 将每行第一个元素写入文件
    FILE* fp = fopen("../media/rope_debug.bin", "wb");
    for (int i = 0; i < M; i++) {
        float val = npu_raw[i * K];
        fwrite(&val, sizeof(float), 1, fp);
    }
    fclose(fp);
    printf("✅ Wrote %d rows to rope_debug.bin\n", M);

    // 复制前10行作为对比
    memcpy(op->npu_output, npu_raw, total_float32 * sizeof(float));
    free(npu_raw);

    printf("CPU first 10: ");
    for (int i = 0; i < 10; i++) printf("%f ", op->cpu_output[i * K]);
    printf("\nNPU first 10: ");
    for (int i = 0; i < 10; i++) printf("%f ", op->npu_output[i * K]);
    printf("\n");

    rknn_destroy_mem(rope_ctx, xm);
    rknn_destroy_mem(rope_ctx, cm);
    rknn_destroy_mem(rope_ctx, sm);
    rknn_destroy_mem(rope_ctx, om);
    return 0;
}

void register_rope_operator() {
    OperatorInterface iface = { .init = rope_init, .run_cpu = rope_run_cpu, .run_npu = rope_run_npu };
    register_operator("RoPE", iface);
}
