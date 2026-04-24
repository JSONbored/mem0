from app.utils.categorization import _resolve_categorization_client, get_categories_for_memory


def _fake_session_with_payload(payload):
    class _FakeResult:
        def fetchone(self):
            return (payload,)

    class _FakeSession:
        def execute(self, _query):
            return _FakeResult()

        def close(self):
            return None

    return _FakeSession()


def test_get_categories_for_memory_skips_when_provider_is_not_openai(monkeypatch):
    monkeypatch.setattr(
        "app.utils.categorization.SessionLocal",
        lambda: _fake_session_with_payload(
            {
                "mem0": {
                    "llm": {
                        "provider": "ollama",
                        "config": {
                            "model": "tinyllama:latest",
                            "ollama_base_url": "http://ollama:11434",
                        },
                    }
                }
            }
        ),
    )

    assert get_categories_for_memory("Postgres on Unraid") == []


def test_resolve_categorization_client_uses_db_openai_config(monkeypatch):
    monkeypatch.setattr(
        "app.utils.categorization.SessionLocal",
        lambda: _fake_session_with_payload(
            {
                "mem0": {
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "api_key": "env:OPENAI_API_KEY",
                            "openai_base_url": "https://example.test/v1",
                        },
                    }
                }
            }
        ),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = _resolve_categorization_client()

    assert client is not None
    assert str(client.base_url) == "https://example.test/v1/"


def test_resolve_categorization_client_uses_non_sqlite_database_settings(monkeypatch):
    class _FakeResult:
        def fetchone(self):
            return (
                {
                    "mem0": {
                        "llm": {
                            "provider": "openai",
                            "config": {
                                "api_key": "env:OPENAI_API_KEY",
                                "openai_base_url": "https://postgres-config.test/v1",
                            },
                        }
                    }
                },
            )

    class _FakeSession:
        def execute(self, _query):
            return _FakeResult()

        def close(self):
            return None

    monkeypatch.setattr("app.utils.categorization.SessionLocal", lambda: _FakeSession())
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = _resolve_categorization_client()

    assert client is not None
    assert str(client.base_url) == "https://postgres-config.test/v1/"
