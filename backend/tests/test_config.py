from app.core.config import Settings


def test_settings_load_defaults():
    settings = Settings(
        POSTGRES_DB="test_db",
        POSTGRES_USER="test_user",
        POSTGRES_PASSWORD="test_password"
    )
    assert settings.PROJECT_NAME == "CHAKRA"
    assert settings.POSTGRES_PORT == 5432
    assert "test_db" in settings.postgres_dsn
    