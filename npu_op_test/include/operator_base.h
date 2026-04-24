#ifndef OPERATOR_BASE_H
#define OPERATOR_BASE_H

#include <time.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdarg.h>

#include "rknn_api.h"
#include "rknn_matmul_api.h"

extern char g_media_dir[512];
extern char g_report_content[8192];

typedef struct {
    char name[32];
    int input_dims[4];
    int output_dims[4];
    float* input_data;
    float* weight_data;
    float* cpu_output;
    float* npu_output;
    int input_size;
    int output_size;
    float relative_error;
    bool passed;
    unsigned int seed;      // 新增：记录初始化时的随机种子
} OperatorTest;

typedef struct {
    void (*init)(OperatorTest* op);
    void (*run_cpu)(OperatorTest* op);
    int (*run_npu)(OperatorTest* op, rknn_context ctx);
} OperatorInterface;

void register_operator(const char* name, OperatorInterface iface);
int get_operator_count();
const char* get_operator_name(int idx);
OperatorTest* get_operator_test(int idx);
OperatorInterface* get_operator_interface(int idx);
void free_op_registry();
void register_all_operators();

float calc_relative_error(float* cpu_out, float* npu_out, int size);
int save_float_data_to_media(const char* filename, float* data, int size);
int save_test_report_to_media(const char* report_content);

#endif
