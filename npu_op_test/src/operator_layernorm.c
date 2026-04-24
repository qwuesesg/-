/**
 * @file operator_layernorm.c
 * @brief LayerNorm 算子: (x - mean) / sqrt(var + eps) * gamma + beta
 *        这里实现简化版，无 affine 参数（或默认为1和0）
 */

#include "operator_base.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <arm_neon.h>
#include "rknn_api.h"
#include "rknn_custom_op.h"

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

// ==================== LayerNorm 初始化 ====================
void layernorm_init(OperatorTest* op) {
    // 输入: [batch, seq_len, hidden] 例如 1x12x64
    op->input_dims[0] = 1;
    op->input_dims[1] = 12;
    op->input_dims[2] = 64;
    op->input_dims[3] = 0;
    op->input_size = 1 * 12 * 64 * sizeof(float);
    memcpy(op->output_dims, op->input_dims, sizeof(op->input_dims));
    op->output_size = op->input_size;

    op->input_data  = (float*)malloc(op->input_size);
    op->weight_data = NULL;   // 不使用 gamma/beta 简化
    op->cpu_output  = (float*)malloc(op->output_size);
    op->npu_output  = (float*)malloc(op->output_size);

    unsigned int seed = (unsigned int)time(NULL);
    op->seed = seed;
    srand(seed);

    int elem = op->input_size / sizeof(float);
    for (int i = 0; i < elem; i++) {
        op->input_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

// ==================== CPU 参考计算 ====================
void layernorm_run_cpu(OperatorTest* op) {
    int batch = op->input_dims[0];
    int seq   = op->input_dims[1];
    int hidden = op->input_dims[2];
    float eps = 1e-5f;

    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq; s++) {
            int offset = b * seq * hidden + s * hidden;
            // 计算均值
            float mean = 0.0f;
            for (int i = 0; i < hidden; i++) {
                mean += op->input_data[offset + i];
            }
            mean /= hidden;
            // 计算方差
            float var = 0.0f;
            for (int i = 0; i < hidden; i++) {
                float diff = op->input_data[offset + i] - mean;
                var += diff * diff;
            }
            var = var / hidden;
            // 归一化
            float inv_std = 1.0f / sqrtf(var + eps);
            for (int i = 0; i < hidden; i++) {
                op->cpu_output[offset + i] = (op->input_data[offset + i] - mean) * inv_std;
            }
        }
    }
}

// ==================== 自定义算子 compute 回调 ====================
static int layernorm_compute(rknn_custom_op_context* op_ctx,
                             rknn_custom_op_tensor* inputs, uint32_t n_inputs,
                             rknn_custom_op_tensor* outputs, uint32_t n_outputs) {
    if (n_inputs != 1 || n_outputs != 1) {
        printf("❌ LayerNorm: need 1 input, 1 output\n");
        return -1;
    }
    float* x = (float*)inputs[0].mem.virt_addr;
    float* out = (float*)outputs[0].mem.virt_addr;
    int n_dims = inputs[0].attr.n_dims;
    int last_dim = inputs[0].attr.dims[n_dims - 1];
    int outer_size = 1;
    for (int i = 0; i < n_dims - 1; i++) {
        outer_size *= inputs[0].attr.dims[i];
    }
    float eps = 1e-5f;

    for (int i = 0; i < outer_size; i++) {
        int offset = i * last_dim;
        float mean = 0.0f;
        for (int j = 0; j < last_dim; j++) mean += x[offset + j];
        mean /= last_dim;
        float var = 0.0f;
        for (int j = 0; j < last_dim; j++) {
            float diff = x[offset + j] - mean;
            var += diff * diff;
        }
        var /= last_dim;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int j = 0; j < last_dim; j++) {
            out[offset + j] = (x[offset + j] - mean) * inv_std;
        }
    }
    return 0;
}

static void register_layernorm_custom_op(rknn_context ctx) {
    rknn_custom_op op = {0};
    op.version = 1;
    op.target = RKNN_TARGET_TYPE_CPU;
    strcpy(op.op_type, "LayerNorm");
    op.compute = layernorm_compute;
    int ret = rknn_register_custom_ops(ctx, &op, 1);
    if (ret != 0) {
        printf("❌ LayerNorm: register custom op failed, error %d\n", ret);
    } else {
        printf("✅ LayerNorm: custom op registered\n");
    }
}

// ==================== NPU 推理接口 ====================
int layernorm_run_npu(OperatorTest* op, rknn_context ctx) {
    static int registered = 0;
    if (!registered) {
        register_layernorm_custom_op(ctx);
        registered = 1;
    }

    static rknn_context ln_ctx = 0;
    if (ln_ctx == 0) {
        const char* model_path = "./models/LayerNorm.rknn";
        int ret = rknn_init(&ln_ctx, (void*)model_path, 0, 0, NULL);
        if (ret != 0) {
            printf("❌ LayerNorm: load model failed, error %d\n", ret);
            return ret;
        }
    }

    rknn_input_output_num io_num;
    int ret = rknn_query(ln_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret) return ret;
    if (io_num.n_input != 1 || io_num.n_output != 1) return -1;

    rknn_tensor_attr input_attr, output_attr;
    memset(&input_attr, 0, sizeof(input_attr));
    input_attr.index = 0;
    ret = rknn_query(ln_ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
    if (ret) return ret;
    memset(&output_attr, 0, sizeof(output_attr));
    output_attr.index = 0;
    ret = rknn_query(ln_ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(output_attr));
    if (ret) return ret;

    rknn_tensor_mem* in_mem  = rknn_create_mem(ln_ctx, input_attr.size);
    rknn_tensor_mem* out_mem = rknn_create_mem(ln_ctx, output_attr.size);
    if (!in_mem || !out_mem) {
        ret = -1;
        goto cleanup;
    }

    int elem = op->input_size / sizeof(float);
    uint16_t* in_f16 = (uint16_t*)malloc(input_attr.size);
    if (!in_f16) { ret = -1; goto cleanup; }
    float32_to_float16(op->input_data, in_f16, elem);
    memcpy(in_mem->virt_addr, in_f16, input_attr.size);
    free(in_f16);

    ret = rknn_set_io_mem(ln_ctx, in_mem, &input_attr);
    if (ret) goto cleanup;
    ret = rknn_set_io_mem(ln_ctx, out_mem, &output_attr);
    if (ret) goto cleanup;

    ret = rknn_run(ln_ctx, NULL);
    if (ret) goto cleanup;

    ret = rknn_mem_sync(ln_ctx, out_mem, RKNN_MEMORY_SYNC_FROM_DEVICE);
    if (ret) goto cleanup;

    if (output_attr.type == RKNN_TENSOR_FLOAT16) {
        int out_elem = output_attr.size / sizeof(uint16_t);
        uint16_t* out_half = (uint16_t*)out_mem->virt_addr;
        float16_to_float32(out_half, op->npu_output, out_elem);
    } else {
        memcpy(op->npu_output, out_mem->virt_addr, output_attr.size);
    }

cleanup:
    if (in_mem)  rknn_destroy_mem(ln_ctx, in_mem);
    if (out_mem) rknn_destroy_mem(ln_ctx, out_mem);
    return ret;
}

void register_layernorm_operator() {
    OperatorInterface iface = {
        .init    = layernorm_init,
        .run_cpu = layernorm_run_cpu,
        .run_npu = layernorm_run_npu
    };
    register_operator("LayerNorm", iface);
}
