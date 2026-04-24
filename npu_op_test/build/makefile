# 项目根目录（绝对路径，无需修改）
ROOT_DIR := /home/topeet/npu_op_test
# 源文件目录（实际 .c 文件所在）
SRC_DIR := $(ROOT_DIR)/src
# 头文件目录（实际 .h 文件所在）
INC_DIR := $(ROOT_DIR)/include
# 库文件目录（librknnrt.so 所在）
LIB_DIR := $(ROOT_DIR)/lib
# 交叉编译器（你提供的完整路径）
CC := /usr/local/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc

# 编译选项（仅包含头文件路径和警告优化）
CFLAGS := -Wall -O2 -I$(INC_DIR)

# 链接选项：库路径 + 动态库 + 静态链接数学库
# -Wl,-Bstatic -lm -Wl,-Bdynamic 表示将 libm.a 静态链接，其他库保持动态链接
LDFLAGS := -L$(LIB_DIR) -lrknnrt -lpthread -ldl -Wl,-Bstatic -lm -Wl,-Bdynamic

# ==================== 源文件列表 ====================
SRC_FILES := $(SRC_DIR)/operator_batch_test.c \
             $(SRC_DIR)/operator_base.c \
             $(SRC_DIR)/operator_add.c \
             $(SRC_DIR)/operator_con2vd.c \
             $(SRC_DIR)/operator_matmul.c \
             $(SRC_DIR)/operator_relu.c \
             $(SRC_DIR)/operator_rope.c \
             $(SRC_DIR)/operator_softmax.c \
             $(SRC_DIR)/operator_silu.c \
             $(SRC_DIR)/operator_layernorm.c

# 输出可执行文件路径
TARGET := $(ROOT_DIR)/bin/operator_batch_test

# ==================== 编译规则（命令前必须用 TAB 缩进）====================
all: $(TARGET)
	@echo "✅ 编译成功！可执行文件路径：$(TARGET)"

# 链接生成可执行文件
$(TARGET): $(SRC_FILES)
	@mkdir -p $(ROOT_DIR)/bin
	$(CC) $(SRC_FILES) -o $(TARGET) $(CFLAGS) $(LDFLAGS)

# 清理编译产物
clean:
	@rm -rf $(ROOT_DIR)/bin
	@echo "✅ 编译产物已清理完成！"

# 伪目标（避免和文件重名）
.PHONY: all clean
