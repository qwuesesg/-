#include "operator_base.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
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

// ==================== 加法算子初始化 ====================
void add_init(OperatorTest* op) {
    op->input_dims[0] = 1;
    op->input_dims[1] = 3;
    op->input_dims[2] = 224;
    op->input_dims[3] = 224;
    op->input_size = 1 * 3 * 224 * 224 * sizeof(float);

    memcpy(op->output_dims, op->input_dims, sizeof(op->input_dims));
    op->output_size = op->input_size;

    op->input_data  = (float*)malloc(op->input_size);
    op->weight_data = (float*)malloc(op->input_size);
    op->cpu_output  = (float*)malloc(op->output_size);
    op->npu_output  = (float*)malloc(op->output_size);

    // 使用固定随机种子，确保可复现
    unsigned int seed = 42;
    op->seed = seed;
    srand(seed);

    int elem_count = op->input_size / sizeof(float);
    for (int i = 0; i < elem_count; i++) {
        // 随机数范围 [-1, 1]
        op->input_data[i]  = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        op->weight_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

// ==================== CPU 参考计算 ====================
void add_run_cpu(OperatorTest* op) {
    int elem_count = op->input_size / sizeof(float);
    for (int i = 0; i < elem_count; i++) {
        op->cpu_output[i] = op->input_data[i] + op->weight_data[i];
    }
}

// ==================== NPU 推理实现 ====================
int add_run_npu(OperatorTest* op, rknn_context ctx) {
    rknn_input_output_num io_num;
    int ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != 0) {
        printf("❌ Add：查询输入输出数量失败，错误码：%d\n", ret);
        return ret;
    }
    if (io_num.n_input != 2 || io_num.n_output != 1) {
        printf("❌ Add：输入输出数量不匹配，预期2输入1输出，实际%d输入%d输出\n",
               io_num.n_input, io_num.n_output);
        return -1;
    }

    rknn_tensor_attr input_attrs[2];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < 2; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != 0) {
            printf("❌ Add：查询输入%d属性失败，错误码：%d\n", i, ret);
            return ret;
        }
    }

    rknn_tensor_attr output_attr;
    memset(&output_attr, 0, sizeof(output_attr));
    output_attr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(rknn_tensor_attr));
    if (ret != 0) {
        printf("❌ Add：查询输出属性失败，错误码：%d\n", ret);
        return ret;
    }

    printf("Add: model input size = %u, output size = %u\n",
           input_attrs[0].size, output_attr.size);
    printf("Add: test input size = %d, output size = %d\n",
           op->input_size, op->output_size);

    size_t model_in_size = input_attrs[0].size;
    size_t model_out_size = output_attr.size;

    size_t alloc_in  = (model_in_size < 16) ? 16 : model_in_size;
    size_t alloc_out = (model_out_size < 16) ? 16 : model_out_size;

    rknn_tensor_mem* input_a_mem = rknn_create_mem(ctx, alloc_in);
    rknn_tensor_mem* input_b_mem = rknn_create_mem(ctx, alloc_in);
    rknn_tensor_mem* output_mem   = rknn_create_mem(ctx, alloc_out);
    if (!input_a_mem || !input_b_mem || !output_mem) {
        printf("❌ Add：NPU内存分配失败\n");
        if (input_a_mem) rknn_destroy_mem(ctx, input_a_mem);
        if (input_b_mem) rknn_destroy_mem(ctx, input_b_mem);
        if (output_mem)   rknn_destroy_mem(ctx, output_mem);
        return -1;
    }

    // 显式分配半精度临时缓冲区
    if (model_in_size == op->input_size / 2) {
        int elem = op->input_size / sizeof(float);
        uint16_t* temp_a = (uint16_t*)malloc(model_in_size);
        uint16_t* temp_b = (uint16_t*)malloc(model_in_size);
        if (!temp_a || !temp_b) {
            printf("❌ Add：临时内存分配失败\n");
            free(temp_a); free(temp_b);
            goto cleanup;
        }
        float32_to_float16(op->input_data,  temp_a, elem);
        float32_to_float16(op->weight_data, temp_b, elem);
        memcpy(input_a_mem->virt_addr, temp_a, model_in_size);
        memcpy(input_b_mem->virt_addr, temp_b, model_in_size);
        free(temp_a); free(temp_b);
    } else if (model_in_size == op->input_size) {
        memcpy(input_a_mem->virt_addr, op->input_data,  model_in_size);
        memcpy(input_b_mem->virt_addr, op->weight_data, model_in_size);
    } else {
        printf("⚠️ Add：模型输入大小(%zu)与测试数据大小(%d)不匹配且无法转换\n",
               model_in_size, op->input_size);
        goto cleanup;
    }

    ret = rknn_set_io_mem(ctx, input_a_mem, &input_attrs[0]);
    if (ret != 0) { printf("❌ Add：绑定输入A失败（错误码：%d）\n", ret); goto cleanup; }

    ret = rknn_set_io_mem(ctx, input_b_mem, &input_attrs[1]);
    if (ret != 0) { printf("❌ Add：绑定输入B失败（错误码：%d）\n", ret); goto cleanup; }

    ret = rknn_set_io_mem(ctx, output_mem, &output_attr);
    if (ret != 0) { printf("❌ Add：绑定输出失败（错误码：%d）\n", ret); goto cleanup; }

    ret = rknn_run(ctx, NULL);
    if (ret != 0) {
        printf("❌ Add：NPU执行失败，错误码：%d\n", ret);
        goto cleanup;
    }

    ret = rknn_mem_sync(ctx, output_mem, RKNN_MEMORY_SYNC_FROM_DEVICE);
    if (ret != 0) {
        printf("❌ Add：内存同步失败（错误码：%d）\n", ret);
        goto cleanup;
    }

    // 输出转换：float16 -> float32
    if (output_attr.type == RKNN_TENSOR_FLOAT16) {
        int elem_count = model_out_size / sizeof(uint16_t);
        uint16_t* out_half = (uint16_t*)output_mem->virt_addr;
        float16_to_float32(out_half, op->npu_output, elem_count);
        printf("Add NPU output (first, after conversion): %f\n", op->npu_output[0]);
    } else {
        memcpy(op->npu_output, output_mem->virt_addr, model_out_size);
        printf("Add NPU output (first, float32): %f\n", ((float*)output_mem->virt_addr)[0]);
    }
    printf("Add CPU output (first): %f\n", op->cpu_output[0]);

cleanup:
    rknn_destroy_mem(ctx, input_a_mem);
    rknn_destroy_mem(ctx, input_b_mem);
    rknn_destroy_mem(ctx, output_mem);
    return ret;
}

void register_add_operator() {
    OperatorInterface iface = {
        .init    = add_init,
        .run_cpu = add_run_cpu,
        .run_npu = add_run_npu
    };
    register_operator("Add", iface);
}
