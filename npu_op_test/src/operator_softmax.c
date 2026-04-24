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

// ==================== Softmax 初始化 ====================
void softmax_init(OperatorTest* op) {
    // 输入形状 [batch=1, seq_len=12, head_dim=64]（模拟 attention scores）
    op->input_dims[0] = 1;
    op->input_dims[1] = 12;
    op->input_dims[2] = 64;
    op->input_dims[3] = 0;   // 未使用
    op->input_size = 1 * 12 * 64 * sizeof(float);
    memcpy(op->output_dims, op->input_dims, sizeof(op->input_dims));
    op->output_size = op->input_size;

    op->input_data = (float*)malloc(op->input_size);
    op->weight_data = NULL;    // Softmax 无权重
    op->cpu_output = (float*)malloc(op->output_size);
    op->npu_output = (float*)malloc(op->output_size);

    srand(time(NULL));
    int elem = op->input_size / sizeof(float);
    for (int i = 0; i < elem; i++) {
        // 随机范围 [-2, 2] 产生明显分布
        op->input_data[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
    }
}

// ==================== CPU 参考计算 ====================
void softmax_run_cpu(OperatorTest* op) {
    int batch = op->input_dims[0];
    int rows = op->input_dims[1];
    int cols = op->input_dims[2];
    float* input = op->input_data;
    float* output = op->cpu_output;

    for (int b = 0; b < batch; b++) {
        for (int r = 0; r < rows; r++) {
            int offset = b * rows * cols + r * cols;
            // 找最大值（数值稳定性）
            float max_val = input[offset];
            for (int c = 1; c < cols; c++) {
                if (input[offset + c] > max_val) max_val = input[offset + c];
            }
            // 计算 exp 并求和
            float sum = 0.0f;
            for (int c = 0; c < cols; c++) {
                float exp_val = expf(input[offset + c] - max_val);
                output[offset + c] = exp_val;
                sum += exp_val;
            }
            // 归一化
            for (int c = 0; c < cols; c++) {
                output[offset + c] /= sum;
            }
        }
    }
}

// ==================== NPU 推理实现 ====================
int softmax_run_npu(OperatorTest* op, rknn_context ctx) {
    // 注意：这里 ctx 是主程序传入的，但我们要为 Softmax 独立加载模型
    // 因此不能使用传入的 ctx，而是自己创建一个新的上下文
    static rknn_context softmax_ctx = 0;
    if (softmax_ctx == 0) {
        const char* model_path = "./models/Softmax.rknn";
        int ret = rknn_init(&softmax_ctx, (void*)model_path, 0, 0, NULL);
        if (ret != 0) {
            printf("❌ Softmax: 加载模型失败，错误码：%d\n", ret);
            return ret;
        }
    }

    // 1. 查询输入输出数量
    rknn_input_output_num io_num;
    int ret = rknn_query(softmax_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != 0) {
        printf("❌ Softmax: 查询输入输出数量失败，错误码：%d\n", ret);
        return ret;
    }
    if (io_num.n_input != 1 || io_num.n_output != 1) {
        printf("❌ Softmax: 输入输出数量不匹配，预期1输入1输出\n");
        return -1;
    }

    // 2. 查询输入属性
    rknn_tensor_attr input_attr;
    memset(&input_attr, 0, sizeof(input_attr));
    input_attr.index = 0;
    ret = rknn_query(softmax_ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(rknn_tensor_attr));
    if (ret != 0) {
        printf("❌ Softmax: 查询输入属性失败，错误码：%d\n", ret);
        return ret;
    }

    // 3. 查询输出属性
    rknn_tensor_attr output_attr;
    memset(&output_attr, 0, sizeof(output_attr));
    output_attr.index = 0;
    ret = rknn_query(softmax_ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(rknn_tensor_attr));
    if (ret != 0) {
        printf("❌ Softmax: 查询输出属性失败，错误码：%d\n", ret);
        return ret;
    }

    // 打印调试信息
    printf("Softmax: model input size = %u, output size = %u\n", input_attr.size, output_attr.size);
    printf("Softmax: test input size = %d, output size = %d\n", op->input_size, op->output_size);

    size_t model_in_size = input_attr.size;   // 模型期望的输入字节数（float16）
    size_t model_out_size = output_attr.size; // 模型输出的字节数（可能是 float16）

    // 4. 分配 NPU 内存（对齐到16字节）
    size_t alloc_in  = (model_in_size < 16) ? 16 : model_in_size;
    size_t alloc_out = (model_out_size < 16) ? 16 : model_out_size;
    rknn_tensor_mem* input_mem = rknn_create_mem(softmax_ctx, alloc_in);
    rknn_tensor_mem* output_mem = rknn_create_mem(softmax_ctx, alloc_out);
    if (!input_mem || !output_mem) {
        printf("❌ Softmax: NPU内存分配失败\n");
        if (input_mem) rknn_destroy_mem(softmax_ctx, input_mem);
        if (output_mem) rknn_destroy_mem(softmax_ctx, output_mem);
        return -1;
    }

    // 5. 数据转换：float32 -> float16
    if (model_in_size == op->input_size / 2) {
        int elem = op->input_size / sizeof(float);
        uint16_t* temp = (uint16_t*)malloc(model_in_size);
        if (!temp) {
            printf("❌ Softmax: 临时内存分配失败\n");
            rknn_destroy_mem(softmax_ctx, input_mem);
            rknn_destroy_mem(softmax_ctx, output_mem);
            return -1;
        }
        float32_to_float16(op->input_data, temp, elem);
        memcpy(input_mem->virt_addr, temp, model_in_size);
        free(temp);
    } else if (model_in_size == op->input_size) {
        memcpy(input_mem->virt_addr, op->input_data, model_in_size);
    } else {
        printf("⚠️ Softmax: 模型输入大小(%zu)与测试数据大小(%d)不匹配\n", model_in_size, op->input_size);
        rknn_destroy_mem(softmax_ctx, input_mem);
        rknn_destroy_mem(softmax_ctx, output_mem);
        return -1;
    }

    // 6. 绑定内存
    ret = rknn_set_io_mem(softmax_ctx, input_mem, &input_attr);
    if (ret != 0) { printf("❌ Softmax: 绑定输入失败（错误码：%d）\n", ret); goto cleanup; }

    ret = rknn_set_io_mem(softmax_ctx, output_mem, &output_attr);
    if (ret != 0) { printf("❌ Softmax: 绑定输出失败（错误码：%d）\n", ret); goto cleanup; }

    // 7. 执行推理
    ret = rknn_run(softmax_ctx, NULL);
    if (ret != 0) {
        printf("❌ Softmax: NPU执行失败，错误码：%d\n", ret);
        goto cleanup;
    }

    // 8. 同步缓存
    ret = rknn_mem_sync(softmax_ctx, output_mem, RKNN_MEMORY_SYNC_FROM_DEVICE);
    if (ret != 0) {
        printf("❌ Softmax: 内存同步失败（错误码：%d）\n", ret);
        goto cleanup;
    }

    // 9. 输出转换：如果模型输出是 float16，转回 float32
    if (output_attr.type == RKNN_TENSOR_FLOAT16) {
        int elem_count = model_out_size / sizeof(uint16_t);
        uint16_t* out_half = (uint16_t*)output_mem->virt_addr;
        float16_to_float32(out_half, op->npu_output, elem_count);
        printf("Softmax NPU output (first): %f\n", op->npu_output[0]);
    } else {
        memcpy(op->npu_output, output_mem->virt_addr, model_out_size);
        printf("Softmax NPU output (first): %f\n", ((float*)output_mem->virt_addr)[0]);
    }
    printf("Softmax CPU output (first): %f\n", op->cpu_output[0]);

cleanup:
    rknn_destroy_mem(softmax_ctx, input_mem);
    rknn_destroy_mem(softmax_ctx, output_mem);
    return ret;
}

// ==================== 注册 Softmax 算子 ====================
void register_softmax_operator() {
    OperatorInterface iface = {
        .init    = softmax_init,
        .run_cpu = softmax_run_cpu,
        .run_npu = softmax_run_npu
    };
    register_operator("Softmax", iface);
}
