/**
 * @file operator_silu.c
 * @brief SiLU (Swish) 激活函数: x * sigmoid(x)
 * @note 采用标准 RKNN 推理流程，与 Softmax、ReLU 等一致
 */

#include "operator_base.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <arm_neon.h>
#include "rknn_api.h"
#include "rknn_matmul_api.h"

// ==================== 数据转换函数 ====================
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

// ==================== SiLU 初始化 ====================
void silu_init(OperatorTest* op) {
    // 形状：1x12x64
    op->input_dims[0] = 1;
    op->input_dims[1] = 12;
    op->input_dims[2] = 64;
    op->input_dims[3] = 0;
    op->input_size = 1 * 12 * 64 * sizeof(float);
    memcpy(op->output_dims, op->input_dims, sizeof(op->input_dims));
    op->output_size = op->input_size;

    op->input_data  = (float*)malloc(op->input_size);
    op->weight_data = NULL;
    op->cpu_output  = (float*)malloc(op->output_size);
    op->npu_output  = (float*)malloc(op->output_size);

    unsigned int seed = (unsigned int)time(NULL);
    op->seed = seed;
    srand(seed);

    int elem = op->input_size / sizeof(float);
    for (int i = 0; i < elem; i++) {
        op->input_data[i] = ((float)rand() / RAND_MAX) * 6.0f - 3.0f;  // [-3,3]
    }
}

// ==================== CPU 参考计算 ====================
void silu_run_cpu(OperatorTest* op) {
    int elem = op->input_size / sizeof(float);
    for (int i = 0; i < elem; i++) {
        float x = op->input_data[i];
        op->cpu_output[i] = x / (1.0f + expf(-x));
    }
}

// ==================== NPU 推理实现 ====================
int silu_run_npu(OperatorTest* op, rknn_context ctx) {
    // 为 SiLU 独立加载模型
    static rknn_context silu_ctx = 0;
    if (silu_ctx == 0) {
        const char* model_path = "./models/SiLU.rknn";
        int ret = rknn_init(&silu_ctx, (void*)model_path, 0, 0, NULL);
        if (ret != 0) {
            printf("❌ SiLU: 加载模型失败，错误码：%d\n", ret);
            return ret;
        }
    }

    // 1. 查询输入输出数量
    rknn_input_output_num io_num;
    int ret = rknn_query(silu_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != 0) {
        printf("❌ SiLU: 查询输入输出数量失败，错误码：%d\n", ret);
        return ret;
    }
    if (io_num.n_input != 1 || io_num.n_output != 1) {
        printf("❌ SiLU: 输入输出数量不匹配，预期1输入1输出\n");
        return -1;
    }

    // 2. 查询输入属性
    rknn_tensor_attr input_attr;
    memset(&input_attr, 0, sizeof(input_attr));
    input_attr.index = 0;
    ret = rknn_query(silu_ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(rknn_tensor_attr));
    if (ret != 0) {
        printf("❌ SiLU: 查询输入属性失败，错误码：%d\n", ret);
        return ret;
    }

    // 3. 查询输出属性
    rknn_tensor_attr output_attr;
    memset(&output_attr, 0, sizeof(output_attr));
    output_attr.index = 0;
    ret = rknn_query(silu_ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(rknn_tensor_attr));
    if (ret != 0) {
        printf("❌ SiLU: 查询输出属性失败，错误码：%d\n", ret);
        return ret;
    }

    // 打印调试信息
    printf("SiLU: model input size = %u, output size = %u\n", input_attr.size, output_attr.size);
    printf("SiLU: test input size = %d, output size = %d\n", op->input_size, op->output_size);

    size_t model_in_size = input_attr.size;
    size_t model_out_size = output_attr.size;

    // 4. 分配 NPU 内存
    rknn_tensor_mem* input_mem = rknn_create_mem(silu_ctx, model_in_size);
    rknn_tensor_mem* output_mem = rknn_create_mem(silu_ctx, model_out_size);
    if (!input_mem || !output_mem) {
        printf("❌ SiLU: NPU内存分配失败\n");
        if (input_mem) rknn_destroy_mem(silu_ctx, input_mem);
        if (output_mem) rknn_destroy_mem(silu_ctx, output_mem);
        return -1;
    }

    // 5. 数据转换：float32 -> float16
    if (model_in_size == op->input_size / 2) {
        int elem = op->input_size / sizeof(float);
        uint16_t* temp = (uint16_t*)malloc(model_in_size);
        if (!temp) {
            printf("❌ SiLU: 临时内存分配失败\n");
            rknn_destroy_mem(silu_ctx, input_mem);
            rknn_destroy_mem(silu_ctx, output_mem);
            return -1;
        }
        float32_to_float16(op->input_data, temp, elem);
        memcpy(input_mem->virt_addr, temp, model_in_size);
        free(temp);
    } else if (model_in_size == op->input_size) {
        memcpy(input_mem->virt_addr, op->input_data, model_in_size);
    } else {
        printf("⚠️ SiLU: 模型输入大小(%zu)与测试数据大小(%d)不匹配\n", model_in_size, op->input_size);
        rknn_destroy_mem(silu_ctx, input_mem);
        rknn_destroy_mem(silu_ctx, output_mem);
        return -1;
    }

    // 6. 绑定内存
    ret = rknn_set_io_mem(silu_ctx, input_mem, &input_attr);
    if (ret != 0) { printf("❌ SiLU: 绑定输入失败（错误码：%d）\n", ret); goto cleanup; }

    ret = rknn_set_io_mem(silu_ctx, output_mem, &output_attr);
    if (ret != 0) { printf("❌ SiLU: 绑定输出失败（错误码：%d）\n", ret); goto cleanup; }

    // 7. 执行推理
    ret = rknn_run(silu_ctx, NULL);
    if (ret != 0) {
        printf("❌ SiLU: NPU执行失败，错误码：%d\n", ret);
        goto cleanup;
    }

    // 8. 同步缓存
    ret = rknn_mem_sync(silu_ctx, output_mem, RKNN_MEMORY_SYNC_FROM_DEVICE);
    if (ret != 0) {
        printf("❌ SiLU: 内存同步失败（错误码：%d）\n", ret);
        goto cleanup;
    }

    // 9. 输出转换
    if (output_attr.type == RKNN_TENSOR_FLOAT16) {
        int elem_count = model_out_size / sizeof(uint16_t);
        uint16_t* out_half = (uint16_t*)output_mem->virt_addr;
        float16_to_float32(out_half, op->npu_output, elem_count);
        printf("SiLU NPU output (first): %f\n", op->npu_output[0]);
    } else {
        memcpy(op->npu_output, output_mem->virt_addr, model_out_size);
        printf("SiLU NPU output (first): %f\n", ((float*)output_mem->virt_addr)[0]);
    }
    printf("SiLU CPU output (first): %f\n", op->cpu_output[0]);

cleanup:
    rknn_destroy_mem(silu_ctx, input_mem);
    rknn_destroy_mem(silu_ctx, output_mem);
    return ret;
}

// ==================== 注册 SiLU 算子 ====================
void register_silu_operator() {
    OperatorInterface iface = {
        .init    = silu_init,
        .run_cpu = silu_run_cpu,
        .run_npu = silu_run_npu
    };
    register_operator("SiLU", iface);
}
