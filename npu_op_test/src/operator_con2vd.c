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

void conv2d_init(OperatorTest* op) {
    op->input_dims[0] = 1;
    op->input_dims[1] = 224;
    op->input_dims[2] = 224;
    op->input_dims[3] = 3;
    op->input_size = 1 * 224 * 224 * 3 * sizeof(float);

    int out_c = 16;
    int kernel_h = 3, kernel_w = 3;
    int stride_h = 1, stride_w = 1;
    int pad = 1;

    op->output_dims[0] = 1;
    op->output_dims[1] = (op->input_dims[1] + 2*pad - kernel_h) / stride_h + 1;
    op->output_dims[2] = (op->input_dims[2] + 2*pad - kernel_w) / stride_w + 1;
    op->output_dims[3] = out_c;
    op->output_size = op->output_dims[0] * op->output_dims[1] * op->output_dims[2] * op->output_dims[3] * sizeof(float);

    int weight_size = out_c * 3 * kernel_h * kernel_w * sizeof(float);
    op->weight_data = (float*)malloc(weight_size);
    op->input_data = (float*)malloc(op->input_size);
    op->cpu_output = (float*)malloc(op->output_size);
    op->npu_output = (float*)malloc(op->output_size);

    unsigned int seed = 42;
    op->seed = seed;
    srand(seed);

    int input_elem = op->input_size / sizeof(float);
    for (int i = 0; i < input_elem; i++) {
        op->input_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    int weight_elem = weight_size / sizeof(float);
    for (int i = 0; i < weight_elem; i++) {
        op->weight_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

void conv2d_run_cpu(OperatorTest* op) {
    int batch = op->input_dims[0];
    int in_h = op->input_dims[1];
    int in_w = op->input_dims[2];
    int in_c = op->input_dims[3];
    int out_h = op->output_dims[1];
    int out_w = op->output_dims[2];
    int out_c = op->output_dims[3];
    int kernel_h = 3, kernel_w = 3, pad = 1, stride = 1;

    memset(op->cpu_output, 0, op->output_size);

    for (int n = 0; n < batch; n++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                for (int oc = 0; oc < out_c; oc++) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < in_c; ic++) {
                        for (int kh = 0; kh < kernel_h; kh++) {
                            for (int kw = 0; kw < kernel_w; kw++) {
                                int ih = oh * stride - pad + kh;
                                int iw = ow * stride - pad + kw;
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    int in_idx = n * in_h * in_w * in_c + ih * in_w * in_c + iw * in_c + ic;
                                    int w_idx = oc * in_c * kernel_h * kernel_w + ic * kernel_h * kernel_w + kh * kernel_w + kw;
                                    sum += op->input_data[in_idx] * op->weight_data[w_idx];
                                }
                            }
                        }
                    }
                    int out_idx = n * out_h * out_w * out_c + oh * out_w * out_c + ow * out_c + oc;
                    op->cpu_output[out_idx] = sum;
                }
            }
        }
    }
}

int conv2d_run_npu(OperatorTest* op, rknn_context ctx) {
    printf("DEBUG: conv2d_run_npu start\n");

    rknn_input_output_num* io_num = (rknn_input_output_num*)calloc(1, sizeof(rknn_input_output_num));
    if (!io_num) { printf("❌ Conv2D：分配 io_num 失败\n"); return -1; }
    int ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, io_num, sizeof(rknn_input_output_num));
    if (ret != 0) {
        printf("❌ Conv2D：查询输入输出数量失败，错误码：%d\n", ret);
        free(io_num);
        return ret;
    }
    printf("DEBUG: io_num->n_input=%d, n_output=%d\n", io_num->n_input, io_num->n_output);
    if (io_num->n_input != 2 || io_num->n_output != 1) {
        printf("❌ Conv2D：输入输出数量不匹配，预期2输入1输出，实际%d输入%d输出\n", io_num->n_input, io_num->n_output);
        free(io_num);
        return -1;
    }

    rknn_tensor_attr* input_attrs = (rknn_tensor_attr*)calloc(2, sizeof(rknn_tensor_attr));
    if (!input_attrs) { free(io_num); printf("❌ Conv2D：分配 input_attrs 失败\n"); return -1; }
    for (int i = 0; i < 2; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != 0) {
            printf("❌ Conv2D：查询输入%d属性失败，错误码：%d\n", i, ret);
            free(input_attrs); free(io_num);
            return ret;
        }
        printf("DEBUG: Input %d size=%u, fmt=%d, type=%d\n", i, input_attrs[i].size, input_attrs[i].fmt, input_attrs[i].type);
    }

    rknn_tensor_attr* output_attr = (rknn_tensor_attr*)calloc(1, sizeof(rknn_tensor_attr));
    if (!output_attr) { free(input_attrs); free(io_num); printf("❌ Conv2D：分配 output_attr 失败\n"); return -1; }
    output_attr->index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, output_attr, sizeof(rknn_tensor_attr));
    if (ret != 0) {
        printf("❌ Conv2D：查询输出属性失败，错误码：%d\n", ret);
        free(output_attr); free(input_attrs); free(io_num);
        return ret;
    }
    printf("DEBUG: Output size=%u, fmt=%d, type=%d\n", output_attr->size, output_attr->fmt, output_attr->type);

    size_t model_in_size_img = input_attrs[0].size;
    size_t model_in_size_kernel = input_attrs[1].size;
    size_t model_out_size = output_attr->size;

    printf("Conv2D: model input sizes = %zu (image), %zu (kernel), output size = %zu\n",
           model_in_size_img, model_in_size_kernel, model_out_size);
    printf("Conv2D: test input size = %d, weight size = %d, output size = %d\n",
           op->input_size, 16*3*3*3*4, op->output_size);

    rknn_tensor_mem* img_mem = rknn_create_mem(ctx, model_in_size_img);
    rknn_tensor_mem* kernel_mem = rknn_create_mem(ctx, model_in_size_kernel);
    rknn_tensor_mem* output_mem = rknn_create_mem(ctx, model_out_size);
    if (!img_mem || !kernel_mem || !output_mem) {
        printf("❌ Conv2D：NPU内存分配失败\n");
        if (img_mem) rknn_destroy_mem(ctx, img_mem);
        if (kernel_mem) rknn_destroy_mem(ctx, kernel_mem);
        if (output_mem) rknn_destroy_mem(ctx, output_mem);
        free(output_attr); free(input_attrs); free(io_num);
        return -1;
    }

    // 图像数据转换为半精度
    if (model_in_size_img == op->input_size / 2) {
        int elem = op->input_size / sizeof(float);
        uint16_t* temp = (uint16_t*)malloc(model_in_size_img);
        if (!temp) { printf("❌ Conv2D：图像临时内存分配失败\n"); goto cleanup; }
        float32_to_float16(op->input_data, temp, elem);
        memcpy(img_mem->virt_addr, temp, model_in_size_img);
        free(temp);
    } else if (model_in_size_img == op->input_size) {
        memcpy(img_mem->virt_addr, op->input_data, model_in_size_img);
    } else {
        printf("⚠️ Conv2D：图像输入大小不匹配\n"); goto cleanup;
    }

    // 权重数据：重排 + 半精度转换
    int weight_elem = 16 * 3 * 3 * 3;
    size_t weight_bytes_f16 = weight_elem * sizeof(uint16_t);
    uint16_t* temp_kernel = (uint16_t*)malloc(weight_bytes_f16);
    if (!temp_kernel) { printf("❌ Conv2D：权重临时内存分配失败\n"); goto cleanup; }

    float* reordered = (float*)malloc(weight_elem * sizeof(float));
    if (!reordered) { free(temp_kernel); printf("❌ Conv2D：权重重排内存失败\n"); goto cleanup; }

    int out_c = 16, in_c = 3, kh = 3, kw = 3;
    for (int oc = 0; oc < out_c; oc++) {
        for (int ic = 0; ic < in_c; ic++) {
            for (int i = 0; i < kh; i++) {
                for (int j = 0; j < kw; j++) {
                    int cpu_idx = oc * in_c * kh * kw + ic * kh * kw + i * kw + j;
                    int model_idx = i * kw * in_c * out_c + j * in_c * out_c + ic * out_c + oc;
                    reordered[model_idx] = op->weight_data[cpu_idx];
                }
            }
        }
    }
    float32_to_float16(reordered, temp_kernel, weight_elem);
    memcpy(kernel_mem->virt_addr, temp_kernel, weight_bytes_f16);
    free(reordered);
    free(temp_kernel);

    ret = rknn_set_io_mem(ctx, img_mem, &input_attrs[0]);
    if (ret != 0) { printf("❌ Conv2D：绑定图像输入失败（错误码：%d）\n", ret); goto cleanup; }

    ret = rknn_set_io_mem(ctx, kernel_mem, &input_attrs[1]);
    if (ret != 0) { printf("❌ Conv2D：绑定权重输入失败（错误码：%d）\n", ret); goto cleanup; }

    ret = rknn_set_io_mem(ctx, output_mem, output_attr);
    if (ret != 0) { printf("❌ Conv2D：绑定输出失败（错误码：%d）\n", ret); goto cleanup; }

    ret = rknn_run(ctx, NULL);
    if (ret != 0) { printf("❌ Conv2D：NPU执行失败，错误码：%d\n", ret); goto cleanup; }

    ret = rknn_mem_sync(ctx, output_mem, RKNN_MEMORY_SYNC_FROM_DEVICE);
    if (ret != 0) { printf("❌ Conv2D：内存同步失败，错误码：%d\n", ret); goto cleanup; }

    if (output_attr->type == RKNN_TENSOR_FLOAT16) {
        int elem_count = model_out_size / sizeof(uint16_t);
        uint16_t* out_half = (uint16_t*)output_mem->virt_addr;
        float16_to_float32(out_half, op->npu_output, elem_count);
        printf("Conv2D NPU output (first, after conversion): %f\n", op->npu_output[0]);
    } else {
        memcpy(op->npu_output, output_mem->virt_addr, model_out_size);
        printf("Conv2D NPU output (first, float32): %f\n", ((float*)output_mem->virt_addr)[0]);
    }
    printf("Conv2D CPU output (first): %f\n", op->cpu_output[0]);

cleanup:
    if (img_mem) rknn_destroy_mem(ctx, img_mem);
    if (kernel_mem) rknn_destroy_mem(ctx, kernel_mem);
    if (output_mem) rknn_destroy_mem(ctx, output_mem);
    free(output_attr);
    free(input_attrs);
    free(io_num);
    return ret;
}

void register_conv2d_operator() {
    OperatorInterface iface = {
        .init = conv2d_init,
        .run_cpu = conv2d_run_cpu,
        .run_npu = conv2d_run_npu
    };
    register_operator("Conv2D", iface);
}
