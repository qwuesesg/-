#include "operator_base.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
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

void relu_init(OperatorTest* op) {
    op->input_dims[0] = 1;
    op->input_dims[1] = 3;
    op->input_dims[2] = 224;
    op->input_dims[3] = 224;
    op->input_size = 1 * 3 * 224 * 224 * sizeof(float);
    memcpy(op->output_dims, op->input_dims, sizeof(op->input_dims));
    op->output_size = op->input_size;

    op->input_data = (float*)malloc(op->input_size);
    op->weight_data = NULL;
    op->cpu_output = (float*)malloc(op->output_size);
    op->npu_output = (float*)malloc(op->output_size);

    unsigned int seed = 42;
    op->seed = seed;
    srand(seed);

    for (int i = 0; i < op->input_size / sizeof(float); i++) {
        op->input_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

void relu_run_cpu(OperatorTest* op) {
    int data_size = op->input_size / sizeof(float);
    for (int i = 0; i < data_size; i++) {
        op->cpu_output[i] = op->input_data[i] > 0 ? op->input_data[i] : 0.0f;
    }
}

int relu_run_npu(OperatorTest* op, rknn_context ctx) {
    rknn_input_output_num io_num;
    int ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != 0) {
        printf("❌ ReLU：查询输入输出数量失败，错误码：%d\n", ret);
        return ret;
    }
    if (io_num.n_input != 1 || io_num.n_output != 1) {
        printf("❌ ReLU：输入输出数量不匹配，预期1输入1输出，实际%d输入%d输出\n", io_num.n_input, io_num.n_output);
        return -1;
    }

    rknn_tensor_attr input_attr;
    memset(&input_attr, 0, sizeof(input_attr));
    input_attr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(rknn_tensor_attr));
    if (ret != 0) {
        printf("❌ ReLU：查询输入属性失败，错误码：%d\n", ret);
        return ret;
    }

    rknn_tensor_attr output_attr;
    memset(&output_attr, 0, sizeof(output_attr));
    output_attr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(rknn_tensor_attr));
    if (ret != 0) {
        printf("❌ ReLU：查询输出属性失败，错误码：%d\n", ret);
        return ret;
    }

    printf("ReLU: model input size = %u, output size = %u\n", input_attr.size, output_attr.size);
    printf("ReLU: test input size = %d, output size = %d\n", op->input_size, op->output_size);

    size_t model_in_size = input_attr.size;
    size_t model_out_size = output_attr.size;

    size_t alloc_in = (model_in_size < 16) ? 16 : model_in_size;
    size_t alloc_out = (model_out_size < 16) ? 16 : model_out_size;

    rknn_tensor_mem* input_mem = rknn_create_mem(ctx, alloc_in);
    rknn_tensor_mem* output_mem = rknn_create_mem(ctx, alloc_out);
    if (!input_mem || !output_mem) {
        printf("❌ ReLU：NPU内存分配失败\n");
        if (input_mem) rknn_destroy_mem(ctx, input_mem);
        if (output_mem) rknn_destroy_mem(ctx, output_mem);
        return -1;
    }

    if (model_in_size == op->input_size / 2) {
        int elem = op->input_size / sizeof(float);
        uint16_t* temp = (uint16_t*)malloc(model_in_size);
        if (!temp) {
            printf("❌ ReLU：临时内存分配失败\n");
            rknn_destroy_mem(ctx, input_mem);
            rknn_destroy_mem(ctx, output_mem);
            return -1;
        }
        float32_to_float16(op->input_data, temp, elem);
        memcpy(input_mem->virt_addr, temp, model_in_size);
        free(temp);
    } else if (model_in_size == op->input_size) {
        memcpy(input_mem->virt_addr, op->input_data, model_in_size);
    } else {
        printf("⚠️ ReLU：模型输入大小(%zu)与测试数据大小(%d)不匹配且无法转换\n", model_in_size, op->input_size);
        rknn_destroy_mem(ctx, input_mem);
        rknn_destroy_mem(ctx, output_mem);
        return -1;
    }

    ret = rknn_set_io_mem(ctx, input_mem, &input_attr);
    if (ret != 0) { printf("❌ ReLU：绑定输入失败（错误码：%d）\n", ret); goto cleanup; }

    ret = rknn_set_io_mem(ctx, output_mem, &output_attr);
    if (ret != 0) { printf("❌ ReLU：绑定输出失败（错误码：%d）\n", ret); goto cleanup; }

    ret = rknn_run(ctx, NULL);
    if (ret != 0) {
        printf("❌ ReLU：NPU执行失败，错误码：%d\n", ret);
        goto cleanup;
    }

    ret = rknn_mem_sync(ctx, output_mem, RKNN_MEMORY_SYNC_FROM_DEVICE);
    if (ret != 0) { printf("❌ ReLU：内存同步失败（错误码：%d）\n", ret); goto cleanup; }

    if (output_attr.type == RKNN_TENSOR_FLOAT16) {
        int elem_count = model_out_size / sizeof(uint16_t);
        uint16_t* out_half = (uint16_t*)output_mem->virt_addr;
        float16_to_float32(out_half, op->npu_output, elem_count);
        printf("ReLU NPU output (first, after conversion): %f\n", op->npu_output[0]);
    } else {
        memcpy(op->npu_output, output_mem->virt_addr, model_out_size);
        printf("ReLU NPU output (first, float32): %f\n", ((float*)output_mem->virt_addr)[0]);
    }
    printf("ReLU CPU output (first): %f\n", op->cpu_output[0]);

cleanup:
    rknn_destroy_mem(ctx, input_mem);
    rknn_destroy_mem(ctx, output_mem);
    return ret;
}

void register_relu_operator() {
    OperatorInterface iface = {
        .init = relu_init,
        .run_cpu = relu_run_cpu,
        .run_npu = relu_run_npu
    };
    register_operator("ReLU", iface);
}
