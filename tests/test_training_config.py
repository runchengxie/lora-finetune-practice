from fine_tuning_practice import TrainSettings, build_peft_config, build_sft_config


def test_peft_config_defaults():
    settings = TrainSettings()
    config = build_peft_config(settings)
    assert config.r == settings.lora_r
    assert config.lora_alpha == settings.lora_alpha
    assert config.lora_dropout == settings.lora_dropout
    assert config.bias == "none"
    assert config.target_modules == "all-linear"
    assert config.task_type == "CAUSAL_LM"


def test_sft_config_cpu_defaults():
    settings = TrainSettings()
    config = build_sft_config(settings, "cpu")
    assert config.bf16 is False
    assert config.fp16 is False
    assert config.optim == "adamw_torch"
    assert config.per_device_train_batch_size == 1
    assert config.gradient_accumulation_steps == 8


def test_sft_config_cuda_fp16():
    settings = TrainSettings()
    config = build_sft_config(settings, "cuda")
    assert config.bf16 is False
    assert config.fp16 is True
