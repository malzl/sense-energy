from sense_energy import config


def test_project_root_contains_expected_dirs():
    assert (config.PROJECT_ROOT / "code").is_dir()
    assert (config.PROJECT_ROOT / "doc").is_dir()
    assert (config.PROJECT_ROOT / "pyproject.toml").is_file()


def test_data_dirs_sit_under_data_dir():
    for path in (config.RAW_DIR, config.INTERIM_DIR, config.PROCESSED_DIR, config.EXTERNAL_DIR):
        assert config.DATA_DIR in path.parents


def test_load_config_reads_yaml():
    cfg = config.load_config("code/configs/features.yaml")
    assert cfg["target"] == "value"
    assert 336 in cfg["target_lags"]
