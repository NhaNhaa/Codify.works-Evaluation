from backend import main as main_module


def test_main_uses_configured_uvicorn_settings(monkeypatch):
    captured = {}

    def fake_run(app_path, host, port, reload, log_level):
        captured["app_path"] = app_path
        captured["host"] = host
        captured["port"] = port
        captured["reload"] = reload
        captured["log_level"] = log_level

    monkeypatch.setattr(main_module, "API_HOST", "0.0.0.0")
    monkeypatch.setattr(main_module, "API_PORT", 8000)
    monkeypatch.setattr(main_module, "API_RELOAD", True)
    monkeypatch.setattr(main_module, "API_LOG_LEVEL", "info")
    monkeypatch.setattr(main_module.uvicorn, "run", fake_run)

    main_module.main()

    assert captured == {
        "app_path": "backend.api:app",
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True,
        "log_level": "info",
    }