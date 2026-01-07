from inference import build_generate_kwargs, resolve_device


def test_generate_kwargs_default_only_max():
    assert build_generate_kwargs(64, None, None) == {"max_new_tokens": 64}


def test_generate_kwargs_temperature():
    kwargs = build_generate_kwargs(64, 0.7, None)
    assert kwargs["max_new_tokens"] == 64
    assert kwargs["temperature"] == 0.7
    assert kwargs["do_sample"] is True
    assert "top_p" not in kwargs


def test_generate_kwargs_top_p():
    kwargs = build_generate_kwargs(64, None, 0.9)
    assert kwargs["max_new_tokens"] == 64
    assert kwargs["top_p"] == 0.9
    assert kwargs["do_sample"] is True
    assert "temperature" not in kwargs


def test_resolve_device_cpu_or_cuda():
    assert resolve_device() in {"cpu", "cuda"}
