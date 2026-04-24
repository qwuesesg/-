# -
本研究旨在 RK3588 边缘计算平台上构建一套白盒化、可扩展的 NPU 算子测试框架，并以此为基础验证大模型（如 LLaMA）中核心算子的 NPU 推理精度。项目采用 TensorFlow + RKNN Toolkit2 完成模型生成，通过 C 语言直接调用 RKNN Runtime API（librknnrt.so）实现板端推理，彻底绕开官方 Python SDK 的黑盒限制，使得内存分配、数据转换、布局处理、缓存同步等环节完全透明可控。
