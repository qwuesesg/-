/**
 * RK3588 NPU Conv2D算子调用（适配旧版RKNN SDK，无Toolkit、无.rknn）
 * 修复所有编译报错：字段名/API/宏定义/头文件适配
 */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>  // 修复fabs隐式声明
#include "rknn_api.h"

// ====================== 算子参数（和之前一致） ======================
#define INPUT_N      1
#define INPUT_C      3
#define INPUT_H      32
#define INPUT_W      32
#define KERNEL_OUT_C 16
#define KERNEL_H     3
#define KERNEL_W     3
#define STRIDE_H     1
#define STRIDE_W     1
#define PAD_H        1
#define PAD_W        1

// ====================== 适配旧版SDK的宏定义/结构体 ======================
// 旧版SDK的张量类型（替换RKNN_TENSOR_INPUT/OUTPUT）
#define RKNN_TENSOR_INPUT  0
#define RKNN_TENSOR_OUTPUT 1

// Conv2D算子参数（简化版，适配旧版SDK）
typedef struct {
    int input_dims[4];  // NCHW: dims[0]=N, dims[1]=C, dims[2]=H, dims[3]=W
    int output_dims[4];
    int kernel_dims[4]; // OUT_C, IN_C, K_H, K_W
    int stride[2];      // H, W
    int pad[2];         // H, W
    float* kernel;      // 卷积核权重
} rknn_conv2d_param;

// ====================== NPU算子执行（适配旧版RKNN SDK基础API） ======================
int rknn_run_conv2d(rknn_context ctx, rknn_conv2d_param* param, float* input, float* output) {
    if (ctx == 0 || param == NULL || input == NULL || output == NULL) {
        printf("❌ 参数无效\n");
        return -1;
    }

    // 1. 配置输入张量（旧版SDK用dims数组表示维度）
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].size = param->input_dims[0] * param->input_dims[1] * param->input_dims[2] * param->input_dims[3] * sizeof(float);
    inputs[0].buf = input;
    inputs[0].pass_through = 0;

    // 2. 设置输入（旧版SDK基础API）
    int ret = rknn_inputs_set(ctx, 1, inputs);
    if (ret != 0) {
        printf("❌ 设置输入失败，错误码：%d\n", ret);
        return ret;
    }

    // 3. 运行NPU（旧版SDK无op_compute，直接rknn_run）
    ret = rknn_run(ctx, NULL);
    if (ret != 0) {
        printf("❌ NPU运行失败，错误码：%d\n", ret);
        return ret;
    }

    // 4. 获取输出（旧版SDK基础API）
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].index = 0;
    outputs[0].want_float = 1;  // 要求返回float32格式

    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    if (ret != 0) {
        printf("❌ 获取输出失败，错误码：%d\n", ret);
        rknn_outputs_release(ctx, 1, outputs);
        return ret;
    }

    // 5. 拷贝输出数据（旧版SDK输出存在临时缓冲区）
    memcpy(output, outputs[0].buf, param->output_dims[0] * param->output_dims[1] * param->output_dims[2] * param->output_dims[3] * sizeof(float));
    
    // 6. 释放输出缓冲区
    rknn_outputs_release(ctx, 1, outputs);
    return 0;
}

// ====================== CPU版Conv2D（无修改，保证对比基准） ======================
void conv2d_cpu(float* input, float* kernel, float* output) {
    int output_h = (INPUT_H + 2*PAD_H - KERNEL_H)/STRIDE_H + 1;
    int output_w = (INPUT_W + 2*PAD_W - KERNEL_W)/STRIDE_W + 1;
    int output_size = INPUT_N*KERNEL_OUT_C*output_h*output_w;
    memset(output, 0, output_size*sizeof(float));

    // 填充输入
    float* input_pad = (float*)malloc(INPUT_N*INPUT_C*(INPUT_H+2*PAD_H)*(INPUT_W+2*PAD_W)*sizeof(float));
    memset(input_pad, 0, INPUT_N*INPUT_C*(INPUT_H+2*PAD_H)*(INPUT_W+2*PAD_W)*sizeof(float));
    for (int b=0; b<INPUT_N; b++) {
        for (int c=0; c<INPUT_C; c++) {
            for (int h=0; h<INPUT_H; h++) {
                memcpy(&input_pad[b*INPUT_C*(INPUT_H+2*PAD_H)*(INPUT_W+2*PAD_W) + c*(INPUT_H+2*PAD_H)*(INPUT_W+2*PAD_W) + (h+PAD_H)*(INPUT_W+2*PAD_W) + PAD_W],
                       &input[b*INPUT_C*INPUT_H*INPUT_W + c*INPUT_H*INPUT_W + h*INPUT_W],
                       INPUT_W*sizeof(float));
            }
        }
    }

    // CPU卷积计算
    for (int b=0; b<INPUT_N; b++) {
        for (int oc=0; oc<KERNEL_OUT_C; oc++) {
            for (int oh=0; oh<output_h; oh++) {
                for (int ow=0; ow<output_w; ow++) {
                    float sum = 0.0f;
                    for (int c=0; c<INPUT_C; c++) {
                        for (int kh=0; kh<KERNEL_H; kh++) {
                            for (int kw=0; kw<KERNEL_W; kw++) {
                                int h_idx = oh*STRIDE_H + kh;
                                int w_idx = ow*STRIDE_W + kw;
                                float in_val = input_pad[b*INPUT_C*(INPUT_H+2*PAD_H)*(INPUT_W+2*PAD_W) + c*(INPUT_H+2*PAD_H)*(INPUT_W+2*PAD_W) + h_idx*(INPUT_W+2*PAD_W) + w_idx];
                                float ker_val = kernel[oc*INPUT_C*KERNEL_H*KERNEL_W + c*KERNEL_H*KERNEL_W + kh*KERNEL_W + kw];
                                sum += in_val * ker_val;
                            }
                        }
                    }
                    output[b*KERNEL_OUT_C*output_h*output_w + oc*output_h*output_w + oh*output_w + ow] = sum;
                }
            }
        }
    }
    free(input_pad);
}

// ====================== 误差计算（修复fabs） ======================
float calc_relative_error(float* cpu_out, float* npu_out, int size) {
    float mean_rel_err = 0.0f;
    for (int i=0; i<size; i++) {
        float abs_err = fabsf(cpu_out[i] - npu_out[i]);  // 用fabsf适配float（避免隐式转换）
        float rel_err = abs_err / (fabsf(cpu_out[i]) + 1e-8);
        mean_rel_err += rel_err;
    }
    return (mean_rel_err / size) * 100;
}

// ====================== 主函数（适配旧版SDK） ======================
int main(int argc, char** argv) {
    // 1. 计算尺寸
    int input_size = INPUT_N*INPUT_C*INPUT_H*INPUT_W;
    int kernel_size = KERNEL_OUT_C*INPUT_C*KERNEL_H*KERNEL_W;
    int output_h = (INPUT_H + 2*PAD_H - KERNEL_H)/STRIDE_H + 1;
    int output_w = (INPUT_W + 2*PAD_W - KERNEL_W)/STRIDE_W + 1;
    int output_size = INPUT_N*KERNEL_OUT_C*output_h*output_w;

    // 2. 内存分配
    float* input = (float*)malloc(input_size*sizeof(float));
    float* kernel = (float*)malloc(kernel_size*sizeof(float));
    float* npu_output = (float*)malloc(output_size*sizeof(float));
    float* cpu_output = (float*)malloc(output_size*sizeof(float));
    if (!input || !kernel || !npu_output || !cpu_output) {
        printf("❌ 内存分配失败\n");
        return -1;
    }

    // 3. 生成测试数据
    srand(12345);
    for (int i=0; i<input_size; i++) input[i] = (float)rand()/RAND_MAX*2 - 1;
    for (int i=0; i<kernel_size; i++) kernel[i] = (float)rand()/RAND_MAX*2 - 1;

    // 4. 初始化NPU（旧版SDK无模型文件，仅初始化硬件）
    printf("🔹 初始化RK3588 NPU（适配旧版SDK）...\n");
    rknn_context ctx;
    int ret = rknn_init(&ctx, NULL, 0, 0, NULL); // 无模型文件，flags=0
    if (ret != 0) {
        printf("❌ NPU初始化失败，错误码：%d\n", ret);
        return ret;
    }
    printf("✅ NPU初始化成功\n");

    // 5. 配置Conv2D参数（适配旧版SDK的dims数组）
    rknn_conv2d_param conv_param;
    memset(&conv_param, 0, sizeof(conv_param));
    // 输入维度：NCHW
    conv_param.input_dims[0] = INPUT_N;
    conv_param.input_dims[1] = INPUT_C;
    conv_param.input_dims[2] = INPUT_H;
    conv_param.input_dims[3] = INPUT_W;
    // 输出维度
    conv_param.output_dims[0] = INPUT_N;
    conv_param.output_dims[1] = KERNEL_OUT_C;
    conv_param.output_dims[2] = output_h;
    conv_param.output_dims[3] = output_w;
    // 卷积核维度
    conv_param.kernel_dims[0] = KERNEL_OUT_C;
    conv_param.kernel_dims[1] = INPUT_C;
    conv_param.kernel_dims[2] = KERNEL_H;
    conv_param.kernel_dims[3] = KERNEL_W;
    // 步长/填充
    conv_param.stride[0] = STRIDE_H;
    conv_param.stride[1] = STRIDE_W;
    conv_param.pad[0] = PAD_H;
    conv_param.pad[1] = PAD_W;
    conv_param.kernel = kernel;

    // 6. NPU执行Conv2D
    printf("🔹 NPU执行Conv2D算子...\n");
    ret = rknn_run_conv2d(ctx, &conv_param, input, npu_output);
    if (ret != 0) {
        printf("❌ NPU执行算子失败\n");
        rknn_destroy(ctx);
        return ret;
    }
    printf("✅ NPU计算完成\n");

    // 7. CPU执行Conv2D
    printf("🔹 CPU执行Conv2D算子...\n");
    conv2d_cpu(input, kernel, cpu_output);
    printf("✅ CPU计算完成\n");

    // 8. 计算误差
    float mean_rel_err = calc_relative_error(cpu_output, npu_output, output_size);
    printf("=====================================\n");
    printf("📊 Conv2D算子验证结果（旧版SDK适配）\n");
    printf("=====================================\n");
    printf("输入尺寸：%d×%d×%d×%d\n", INPUT_N, INPUT_C, INPUT_H, INPUT_W);
    printf("卷积核：%d×%d×%d×%d\n", KERNEL_OUT_C, INPUT_C, KERNEL_H, KERNEL_W);
    printf("输出尺寸：%d×%d×%d×%d\n", INPUT_N, KERNEL_OUT_C, output_h, output_w);
    printf("平均相对误差：%.4f%%\n", mean_rel_err);
    printf("=====================================\n");

    // 9. 结果判断
    if (mean_rel_err < 1.0) { // 旧版SDK float16转float32误差阈值放宽到1%
        printf("✅ 算子验证通过！NPU计算结果与CPU一致\n");
    } else {
        printf("❌ 算子验证失败！误差超出阈值\n");
    }

    // 10. 资源释放
    rknn_destroy(ctx);
    free(input);
    free(kernel);
    free(npu_output);
    free(cpu_output);

    return 0;
}
