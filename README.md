# -
一、项目背景与目标
本研究旨在 RK3588 边缘计算平台上构建一套白盒化、可扩展的 NPU 算子测试框架，并以此为基础验证大模型（如 LLaMA）中核心算子的 NPU 推理精度。项目采用 TensorFlow + RKNN Toolkit2 完成模型生成，通过 C 语言直接调用 RKNN Runtime API（librknnrt.so）实现板端推理，打破黑盒限制，使得内存分配、数据转换、布局处理、缓存同步等环节完全透明可控。

二、当前系统架构与实现
1. 模块化测试框架
公共基础设施（operator_base.c/h）：定义算子测试接口 OperatorInterface、测试数据结构 OperatorTest，实现动态注册链表、相对误差计算、二进制数据持久化。

算子实现模块：每个算子对应一个 .c 文件，实现 init（生成测试数据）、run_cpu（CPU 参考计算）、run_npu（NPU 推理）三个核心函数。

主控程序（operator_batch_test.c）：依次注册所有算子，加载 RKNN 模型，执行计算，计算相对误差，生成测试报告并保存输入/输出/种子数据。

2. 模型生成流程
在 PC 端使用 TensorFlow 构建单算子计算图，冻结为 .pb 文件，再通过 RKNN-Toolkit2（2.3.0 版本）配置 target_platform='rk3588' 并转换为 .rknn 模型。各算子的输入形状和数据布局均根据算子特性精心设计，例如 Conv2D 采用 NHWC [1,224,224,3] 双输入（图像+权重），MatMul 采用 NCHW 抽象表示矩阵维度，Softmax/SiLU/LayerNorm 展平为二维 [1,12,64] 等。

3. NPU 推理标准流程
所有标准算子均遵循统一的十二步调用序列：

rknn_init 加载模型

rknn_query(RKNN_QUERY_IN_OUT_NUM) 获取输入/输出数量

rknn_query(RKNN_QUERY_INPUT_ATTR) 获取各输入张量属性

rknn_query(RKNN_QUERY_OUTPUT_ATTR) 获取输出张量属性

rknn_create_mem 为各张量分配 NPU 内存

若模型要求 float16（model_in_size == op->input_size / 2），则使用 ARM NEON 指令将 float32 输入转换为 float16

memcpy 将转换后数据拷贝至 NPU 内存

rknn_set_io_mem 绑定各输入输出内存

rknn_run 执行推理

rknn_mem_sync(RKNN_MEMORY_SYNC_FROM_DEVICE) 同步缓存

若输出为 float16，调用 float16_to_float32 转回 float32

rknn_destroy_mem 释放内存，rknn_destroy 销毁上下文

4. 特殊机制
Conv2D：支持动态权重传递，CPU 侧权重存储为 [out_c, in_c, kh, kw]，NPU 期望 [kh, kw, in_c, out_c]，在 run_npu 中进行权重重排。

MatMul：采用专用的 rknn_matmul_create / rknn_matmul_set_io_mem / rknn_matmul_run API 实现矩阵乘，并封装为通用接口 rknn_matmul_forward(ctx, A, B, C, M, K, N)，方便后续 LLaMA 框架集成。

Softmax/SiLU/LayerNorm：因 RKNN Toolkit 2.3.0 对高维自定义算子的支持有限，将输入展平为二维，使用标准 TensorFlow 算子组合（如 tf.sigmoid、tf.multiply、tf.nn.softmax、手动实现 LayerNorm）生成模型，板端按上述标准流程推理。

5. 测试结果
已注册 8 个算子，其中 7 个标准算子的 NPU 推理误差均远小于 1% 阈值：

算子	           输入形状	               相对误差     	        备注
Add	          [1,3,224,224]     	       0.1156%	         双输入逐元素加法
ReLU	        [1,3,224,224]	             0.0084%	         简单激活函数，精度损失极低
Conv2D	      [1,224,224,3] (NHWC)	     0.2392%	         包含权重重排与动态权重传递
MatMul	      [1,64,1,128] × [128,64]	   0.1679%           专用矩阵乘 API，输出为 float32
Softmax	      [1,12,64]                  0.0663%	         数值稳定性优化
SiLU	        [1,12,64]	                 0.0771%	         Swish 激活函数
LayerNorm	    [1,12,64]	                 0.1051%	         无 affine 参数的简化版
