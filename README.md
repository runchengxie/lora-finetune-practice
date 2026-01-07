# lora-finetune-practice

一个用于练习 LoRA 微调的最小项目，默认配置偏向低显存（例如 GTX 970 级别）。

## 功能简介

- 基于 `trl` + `peft` 对 `SmolLM2-135M` 做 LoRA 微调
- 默认关闭 bf16、fused optimizer，避免老卡不兼容
- 训练参数以“能跑通”为目标：小 batch、短序列、较高累积步数

## 环境要求

- Python 3.9+（建议 3.10+）
- 可选 GPU（CUDA 能显著加速，老卡只要支持 fp16 也能试）

## 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

如果需要运行测试：

```bash
pip install -e ".[test]"
```

## Hugging Face Token（可选）

公开数据集/模型可以不登录；需要访问私有资源时再配置即可。

1) 复制示例文件并填写 Token：

```bash
cp .env.example .env
cp .envrc.example .envrc
```

2) 在 `.env` 中设置：

```
HF_TOKEN=hf_xxx
```

3) 如果你用 direnv：

```bash
direnv allow
```

## 运行训练

```bash
python fine_tuning_practice.py
```

如果要调整超参，直接修改 `fine_tuning_practice.py` 里的 `TrainSettings` 默认值即可。

## 合并 LoRA Adapter（可选）

将 `TrainSettings.merge_adapter` 设为 `True` 后再次运行，会把 adapter 合并进模型权重并保存到输出目录。

## 简单推理示例（可选）

将 `TrainSettings.run_inference` 设为 `True`，训练后会用内置的几条提示语跑一次推理，便于快速验收效果。

## 运行测试

```bash
./scripts/run_tests.sh
```

## 缓存位置说明

Hugging Face 会缓存模型和数据集，默认位置是：

- `~/.cache/huggingface`

常见的子目录：

- 模型与 tokenizer：`~/.cache/huggingface/hub`
- 数据集：`~/.cache/huggingface/datasets`

你可以用这些环境变量改位置：

- `HF_HOME`：统一缓存根目录
- `HF_HUB_CACHE`：模型缓存目录
- `HF_DATASETS_CACHE`：数据集缓存目录
- `TRANSFORMERS_CACHE`：transformers 缓存目录（旧变量，仍可用）
