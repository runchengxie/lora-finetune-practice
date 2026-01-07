# lora-finetune-practice

一个用于练习 LoRA 微调的最小项目，默认配置偏向低显存。

## 功能简介

* 基于 `trl` + `peft` 对 `SmolLM2-135M` 做 LoRA 微调
* 默认关闭 bf16、fused optimizer，避免老卡不兼容
* 训练参数以“能跑通”为目标：小 batch、短序列、较高累积步数

## 环境要求

* Python 3.9+（建议 3.10+）
* 可选 GPU（CUDA 能显著加速，老卡只要支持 fp16 也能试）
* 设备支持：CPU / CUDA（MPS 不参与）

## 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

如果使用 `uv`：

```bash
uv venv
uv sync
```

如需 CUDA，请先按 PyTorch 官方说明安装对应版本，再执行上面的安装命令。

如果需要运行测试：

```bash
pip install -e ".[test]"
```

使用 `uv` 安装测试依赖：

```bash
uv sync --extra test
```

## 项目结构

* `fine_tuning_practice.py`：训练主脚本
* `inference.py`：推理脚本
* `scripts/upload_lora_to_hub.py`：上传脚本
* `scripts/run_tests.sh`：测试入口
* `tests/`：单元测试
* `outputs/`：训练输出默认目录（自动生成）

## Hugging Face Token（可选）

公开数据集/模型可以不登录；需要访问私有资源时再配置即可。

1) 复制示例文件并填写 Token：

```bash
cp .env.example .env
cp .envrc.example .envrc
```

1) 在 `.env` 中设置：

```
HF_TOKEN=hf_xxx
```

1) 如果你用 direnv：

```bash
direnv allow
```

## 运行训练

```bash
python fine_tuning_practice.py
```

如果你使用 `uv` 管理环境：

```bash
uv run python fine_tuning_practice.py
```

如果要调整超参，直接修改 `fine_tuning_practice.py` 里的 `TrainSettings` 默认值即可。

## 训练产物解读

输出目录里按功能，基本分两条来源：

1. 训练过程中按策略保存的 checkpoint（中间态）
2. 训练结束 `trainer.save_model()` + `tokenizer.save_pretrained(...)` 保存的 最终产物（可用于推理/分享）

输出目录名字来自 `TrainSettings.output_dir`，默认就是 `outputs/SmolLM2-FT-MyDataset`。

### 1) `adapter_model.safetensors`

LoRA 的权重文件，也是这次微调真正训练出来的。

* 什么时候用：推理、分享、部署（LoRA 方案）
* 是否必须：是（LoRA 路径下）

### 2) `adapter_config.json`

LoRA 配置，告诉加载器“这套 adapter 是怎么训练出来的”（例如 `r`、`lora_alpha`、`lora_dropout`、目标模块等）。

* 什么时候用：加载 adapter 时必读
* 是否必须：是

### 3) `checkpoint-*/`

训练过程的检查点（中间态），用于断点续训或回滚对比。

默认参数下粗略是：`train_subset_size=2000`、`per_device_train_batch_size=1`、`gradient_accumulation_steps=8`、`num_train_epochs=1`，
所以会在一个 epoch 结束时生成类似 `checkpoint-250/`（数字会随设置变化）。

* 什么时候用：继续训练、对比不同 step 的效果
* 是否必须：推理不需要；续训才需要

### 4) `training_args.bin`

训练参数的序列化快照（Trainer 的训练配置落盘）。

* 什么时候用：复盘“我当时到底怎么训的”、对齐实验记录
* 是否必须：推理不需要，但强烈建议保留

### 5) Tokenizer 相关（这是一组）

通常会看到：

* `tokenizer.json`
* `tokenizer_config.json`
* `special_tokens_map.json`
* `merges.txt`
* `vocab.json`

这套是 tokenizer 的定义与词表（BPE merges/vocab、特殊 token 映射、配置等）。

* 什么时候用：推理时做分词；分享 adapter 时让目录“自包含”
* 是否必须：严格说 LoRA 推理可以用 base model 的 tokenizer，但为了避免“别人加载时 tokenizer/模板对不上”的坑，建议跟 adapter 一起保留/上传

### 6) `chat_template.jinja`

聊天格式模板（把 user/system/assistant 这些消息结构拼成模型能吃的文本）。

* 什么时候用：做对话式推理时，保证格式一致
* 是否必须：做 chat 模式时非常有价值（尤其之后换环境/换加载方式）

### 7) `README.md`

通常是模型卡/说明文件（有时框架会自动生成，有时是你补充的）。

* 什么时候用：分享、协作、版本说明
* 是否必须：不必须，但很建议保留/上传

### 8) （可选）合并后的完整模型权重

当你把 `TrainSettings.merge_adapter = True` 时，会在输出目录里额外写入合并后的模型权重（例如 `model.safetensors`、`config.json` 等）。

* 什么时候用：想把 LoRA 变成“单体模型”部署
* 是否必须：不是必需，保留 LoRA 也能推理

### 这些东西到底该怎么处理（推理/分享/续训）

只做推理（最常见）：

* `adapter_model.safetensors`
* `adapter_config.json`
* tokenizer 那一套（建议带上）
* `chat_template.jinja`（做 chat 建议带上）
* `README.md`（建议带上）

要分享给别人或上 Hub 做版本管理：

* 同“只做推理”，再加上 `training_args.bin`（方便复现）

还要继续训练（续训）：

* checkpoint 必须留（至少留最新那个）
* `training_args.bin` 也该留着

## 合并 LoRA Adapter（可选）

将 `TrainSettings.merge_adapter` 设为 `True` 后再次运行，会把 adapter 合并进模型权重并保存到输出目录。

## 简单推理示例（可选）

将 `TrainSettings.run_inference` 设为 `True`，训练后会用内置的几条提示语跑一次推理，便于快速验收效果。

## 推理（脚本）

使用训练输出目录做推理：

```bash
python inference.py --model-dir outputs/SmolLM2-FT-MyDataset --prompt "What is LoRA?"
```

如果你使用 `uv` 管理环境，建议用 `uv run` 来确保依赖就绪：

```bash
uv run python inference.py --model-dir outputs/SmolLM2-FT-MyDataset --prompt "What is LoRA?"
```

下面的示例同理，前面加 `uv run` 即可。

多条提示语可以重复 `--prompt`：

```bash
python inference.py \
  --model-dir outputs/SmolLM2-FT-MyDataset \
  --prompt "What is LoRA?" \
  --prompt "Explain gradient accumulation briefly."
```

如果目录里缺 tokenizer 文件，可以补 `--base-model`：

```bash
python inference.py --model-dir outputs/SmolLM2-FT-MyDataset --base-model HuggingFaceTB/SmolLM2-135M
```

## 上传到 Hugging Face Hub（脚本）

先确保 `HF_TOKEN` 已配置（见上面的 Token 配置），再执行：

```bash
python scripts/upload_lora_to_hub.py \
  --repo-id your-username/your-repo \
  --local-dir outputs/SmolLM2-FT-MyDataset
```

如果你使用 `uv` 管理环境：

```bash
uv run python scripts/upload_lora_to_hub.py \
  --repo-id your-username/your-repo \
  --local-dir outputs/SmolLM2-FT-MyDataset
```

常用选项：

* `--exclude-checkpoints`：不上传 `checkpoint-*`，节省空间
* `--public`：创建公开仓库（默认私有）
* `--commit-message`：自定义提交说明

## 运行测试

```bash
./scripts/run_tests.sh
```

如果你使用 `uv` 管理环境：

```bash
uv run ./scripts/run_tests.sh
```

## 缓存位置说明

Hugging Face 会缓存模型和数据集，默认位置是：

* `~/.cache/huggingface`

常见的子目录：

* 模型与 tokenizer：`~/.cache/huggingface/hub`
* 数据集：`~/.cache/huggingface/datasets`

你可以用这些环境变量改位置：

* `HF_HOME`：统一缓存根目录
* `HF_HUB_CACHE`：模型缓存目录
* `HF_DATASETS_CACHE`：数据集缓存目录
* `TRANSFORMERS_CACHE`：transformers 缓存目录（旧变量，仍可用）
