#include "operator_base.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>

extern char g_media_dir[512];
extern char g_report_content[8192];

void register_add_operator();
void register_conv2d_operator();
void register_matmul_operator();
void register_relu_operator();
void register_softmax_operator();
void register_silu_operator();
void register_layernorm_operator();
void register_rope_operator();

void register_all_operators() {
    register_add_operator();
    register_conv2d_operator();
    register_matmul_operator();
    register_relu_operator();
    register_softmax_operator();   
    register_silu_operator();      
    register_layernorm_operator(); 
    register_rope_operator();
}

int main() {
    const char* env_media = getenv("MEDIA_DIR");
    if (env_media) {
        snprintf(g_media_dir, sizeof(g_media_dir), "%s", env_media);
    } else {
        snprintf(g_media_dir, sizeof(g_media_dir), "../media/");
    }
    char mkdir_cmd[1024];
    snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", g_media_dir);
    system(mkdir_cmd);
    printf("🔹 创建media目录完成（存放测试数据）：%s\n", g_media_dir);

    register_all_operators();
    printf("✅ 共注册 %d 个算子：", get_operator_count());
    for (int i = 0; i < get_operator_count(); i++) {
        printf(" %s", get_operator_name(i));
    }
    printf("\n\n");

    rknn_context ctx_main;
    const char* rknn_model_path = "/root/npu_test/min_model.rknn";
    int ret = rknn_init(&ctx_main, (void*)rknn_model_path, 0, 0, NULL);
    if (ret != 0) {
        printf("❌ NPU初始化失败，错误码：%d\n", ret);
        free_op_registry();
        return -1;
    }
    printf("✅ NPU主上下文初始化成功\n\n");

    int op_count = get_operator_count();
    int pass_count = 0;
    strcat(g_report_content, "📋 RK3588 NPU算子批量测试报告\n");
    strcat(g_report_content, "=====================================\n");
    time_t now = time(NULL);
    strcat(g_report_content, "测试时间：");
    strcat(g_report_content, ctime(&now));
    strcat(g_report_content, "=====================================\n");

    for (int i = 0; i < op_count; i++) {
        const char* op_name = get_operator_name(i);
        OperatorTest* op = get_operator_test(i);
        OperatorInterface* iface = get_operator_interface(i);
        rknn_context ctx;

        // MatMul 使用主上下文，其他算子各自加载模型
        if (strcmp(op_name, "MatMul") == 0) {
            ctx = ctx_main;
        } else {
            char model_path[256];
            snprintf(model_path, sizeof(model_path), "./models/%s.rknn", op_name);
            ret = rknn_init(&ctx, model_path, 0, 0, NULL);
            if (ret != 0) {
                printf("❌ 加载模型 %s 失败，错误码：%d\n", model_path, ret);
                continue;
            }
        }

        printf("=====================================\n");
        printf("🔹 测试算子：%s\n", op_name);
        strcat(g_report_content, "\n【算子：");
        strcat(g_report_content, op_name);
        strcat(g_report_content, "】\n");

        iface->init(op);
        printf("  ✅ 初始化完成（输入维度：%d x %d x %d x %d，随机种子：%u）\n",
               op->input_dims[0], op->input_dims[1], op->input_dims[2], op->input_dims[3], op->seed);
        char dim_str[128];
        snprintf(dim_str, sizeof(dim_str), "  输入维度：%d x %d x %d x %d，种子：%u\n",
                op->input_dims[0], op->input_dims[1], op->input_dims[2], op->input_dims[3], op->seed);
        strcat(g_report_content, dim_str);

        iface->run_cpu(op);
        printf("  ✅ CPU计算完成\n");
        strcat(g_report_content, "  CPU计算：完成\n");

        ret = iface->run_npu(op, ctx);
        if (ret != 0) {
            printf("  ❌ NPU执行失败！错误码：%d\n", ret);
            strcat(g_report_content, "  NPU执行：失败（错误码：");
            char err_str[32];
            snprintf(err_str, sizeof(err_str), "%d", ret);
            strcat(g_report_content, err_str);
            strcat(g_report_content, "）\n");
            if (strcmp(op_name, "MatMul") != 0) {
                rknn_destroy(ctx);
            }
            continue;
        }
        printf("  ✅ NPU白盒执行完成\n");
        strcat(g_report_content, "  NPU执行：完成\n");

        int data_size = op->output_size / sizeof(float);
        float err = calc_relative_error(op->cpu_output, op->npu_output, data_size);
        op->relative_error = err;
        printf("  📈 平均相对误差：%.4f%%\n", err);

        char err_str[64];
        snprintf(err_str, sizeof(err_str), "  平均相对误差：%.4f%%\n", err);
        strcat(g_report_content, err_str);

        char filename[128];
        snprintf(filename, sizeof(filename), "%s_input.bin", op_name);
        save_float_data_to_media(filename, op->input_data, op->input_size / sizeof(float));

        if (op->weight_data) {
            snprintf(filename, sizeof(filename), "%s_weight.bin", op_name);
            int weight_elem = (op_name[0] == 'C') ? (16 * 3 * 3 * 3) : (op->input_size / sizeof(float));
            save_float_data_to_media(filename, op->weight_data, weight_elem);
        }

        snprintf(filename, sizeof(filename), "%s_cpu_output.bin", op_name);
        save_float_data_to_media(filename, op->cpu_output, data_size);

        snprintf(filename, sizeof(filename), "%s_npu_output.bin", op_name);
        save_float_data_to_media(filename, op->npu_output, data_size);

        // 保存种子信息
        snprintf(filename, sizeof(filename), "%s_seed.txt", op_name);
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", g_media_dir, filename);
        FILE* fp = fopen(path, "w");
        if (fp) {
            fprintf(fp, "%u\n", op->seed);
            fclose(fp);
        }

        if (err < 1.0f) {
            op->passed = true;
            pass_count++;
            printf("  ✅ 测试通过\n");
            strcat(g_report_content, "  状态：通过\n");
        } else {
            op->passed = false;
            printf("  ❌ 测试失败（误差>1%%）\n");
            strcat(g_report_content, "  状态：失败（误差超标）\n");
        }

        // 销毁非共享的上下文
        if (strcmp(op_name, "MatMul") != 0) {
            rknn_destroy(ctx);
        }
    }

    strcat(g_report_content, "\n=====================================\n");
    char summary[256];
    snprintf(summary, sizeof(summary), "📊 测试总结：共 %d 个算子，通过 %d 个，失败 %d 个\n",
             op_count, pass_count, op_count - pass_count);
    strcat(g_report_content, summary);
    printf("\n%s", summary);

    if (save_test_report_to_media(g_report_content) == 0) {
        printf("✅ 测试报告已保存到：%s/test_report.log\n", g_media_dir);
    } else {
        printf("❌ 测试报告保存失败！\n");
    }

    // 释放算子内存
    for (int i = 0; i < op_count; i++) {
        OperatorTest* op = get_operator_test(i);
        if (op->input_data) free(op->input_data);
        if (op->weight_data) free(op->weight_data);
        if (op->cpu_output) free(op->cpu_output);
        if (op->npu_output) free(op->npu_output);
    }
    rknn_destroy(ctx_main);
    free_op_registry();
    return 0;
}
