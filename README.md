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


三、自定义算子 RoPE 的探索与困境
1. 目标与初衷
旋转位置编码（RoPE）是大语言模型 Transformer 架构中的核心组件，但 RK3588 NPU 硬件并不直接支持该算子。根据 RKNN API 参考手册，运行时提供了 rknn_register_custom_ops 接口，允许用户注册自定义算子类型，并在模型推理时由 NPU 调度执行用户编写的 compute 回调函数。本研究希望借此实现 RoPE 的 NPU 框架集成，验证自定义算子机制在 RKNN 生态中的可用性。

2. 已做的工作
板端自定义算子注册：成功编写 rope_compute 回调函数，并通过 rknn_register_custom_ops 注册为 RoPE 类型，回调函数可被正确调用。

CPU 参考实现：用 C 语言实现了完整的 RoPE 计算，包括 cos/sin 频率表生成与复数旋转公式，计算结果作为精度比较的基准。

多轮模型生成尝试：为将 RoPE 算子部署到 NPU 上，尝试了多种模型构建策略：

四维原始模型：用 TensorFlow 标准算子模拟 RoPE（xcos - xsin 等），输入/输出形状为 [1,128,12,64]（batch×seq_len×num_heads×head_dim）。NPU 输出格式为 NCHW（fmt=0），我们试图通过 NCHW→NHWC 转换匹配 CPU 结果，但布局映射始终错误，平均相对误差约 130%。

二维展平模型：将输入展平为 [1536, 64]，完全消除四维布局歧义。该模型在 NPU 上成功运行，且前 10 个输出值与 CPU 结果高度吻合（仅存在 float16 精度损失），然而从第 12 行开始，误差急剧增大，整体相对误差飙升至 500%～1000%。经多种维度重排假设（8 种）验证，均无法使 NPU 输出与 CPU 一致，排除了布局错位的可能性。

强制 float32 尝试：在生成 RKNN 模型时指定 float_dtype='float32'，希望 NPU 内部采用浮点单精度计算，但工具链实际忽略该配置，输出仍为 float16。

直接调用验证：由于无法生成包含自定义 RoPE 节点的 RKNN 模型，我们在测试程序中绕过模型加载，直接手动构造 rknn_custom_op_tensor 结构体并调用 rope_compute，验证了回调函数本身的正确性。

3. 原因分析
造成 RoPE 无法在 NPU 上正确运行的根本原因有两个层面：

工具链层：RKNN Toolkit 2.3.0 不支持生成包含自定义算子的 RKNN 模型。其 config() 接口的 custom_string 参数无法声明自定义算子类型，且 rknn.build() 过程中 ONNX Runtime 会对计算图进行完整性校验，遇到未知算子（RoPE）直接报错，无法导出有效模型。

硬件层：当用标准算子（Mul、Add、Sub 等）模拟 RoPE 时，NPU 内部强制以 float16 进行推理。RoPE 涉及幂运算、三角函数以及乘积累，中间结果极易超出 float16 的表示范围（上限约 65504），导致数值溢出，计算结果严重失真。

