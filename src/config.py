from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    LLM_API_KEY: str
    LLM_BASE_URL: str | None = "https://openrouter.ai/api/v1"


settings = Settings()
