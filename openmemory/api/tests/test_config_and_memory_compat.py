import datetime

import pytest

from app.mcp_server import (
    _list_memories_via_client,
    _score_memory_match,
    _search_memories_from_list,
    _search_memories_via_client,
    client_name_var,
    list_memories,
    search_memory,
    user_id_var,
)
from app.routers.config import (
    ConfigSchema,
    EmbedderConfig,
    EmbedderProvider,
    LLMConfig,
    LLMProvider,
    Mem0Config,
    OpenMemoryConfig,
    get_default_configuration,
    update_configuration,
)
from app.utils.memory import (
    _drop_none_entries,
    _ensure_vector_store_dimensions,
    _repair_qdrant_collection_if_needed,
    get_default_memory_config,
)


class _SearchClientFiltersOnly:
    def search(self, query, *, filters=None):
        assert query == "coffee"
        assert filters == {"user_id": "alice"}
        return {
            "results": [
                {
                    "id": "123",
                    "memory": "Prefers coffee",
                    "metadata": {"hash": "abc"},
                    "score": 0.91,
                }
            ]
        }


class _SearchClientLegacyOnly:
    def search(self, query, *, filters=None, user_id=None):
        if filters is not None:
            raise TypeError("filters is not supported")
        assert query == "coffee"
        assert user_id == "alice"
        return [{"id": "456", "data": "Legacy coffee memory", "hash": "def"}]


class _ListClientFiltersOnly:
    def get_all(self, *, filters=None):
        assert filters == {"user_id": "alice"}
        return {"results": [{"id": "789", "memory": "Tea note"}]}


class _ListClientLegacyOnly:
    def get_all(self, *, filters=None, user_id=None):
        if filters is not None:
            raise ValueError("filters not supported")
        assert user_id == "alice"
        return [{"id": "987", "data": "Legacy tea note"}]


class _SearchClientUnscopedFallback:
    def search(self, query, *, filters=None, user_id=None):
        if filters is not None:
            raise ValueError("filters require unavailable stopwords backend")
        if user_id is not None:
            raise ValueError("user_id is not supported")
        assert query == "coffee"
        return [{"id": "654", "data": "Unscoped coffee memory"}]


class _ListClientUnscopedFallback:
    def get_all(self, *, filters=None, user_id=None):
        if filters is not None:
            raise ValueError("filters are not supported")
        if user_id is not None:
            raise ValueError("user_id is not supported")
        return [{"id": "321", "data": "Unscoped tea note"}]


class _SearchFallsBackToListClient:
    def search(self, query, *, filters=None, user_id=None):
        if filters is not None:
            raise ValueError("filters require unavailable stopwords backend")
        if user_id is not None:
            raise ValueError("user_id is not supported")
        raise ValueError("filters must contain at least one of: user_id, agent_id, run_id")

    def get_all(self, *, filters=None, user_id=None):
        if filters is not None:
            assert filters == {"user_id": "alice"}
        elif user_id is not None:
            assert user_id == "alice"

        return {
            "results": [
                {"id": "1", "memory": "backend smoke redis memory on Unraid with Postgres"},
                {"id": "2", "memory": "completely unrelated note"},
            ]
        }


class _SearchReturnsNoResultsClient:
    def search(self, query, *, filters=None, user_id=None):
        return {"results": []}

    def get_all(self, *, filters=None, user_id=None):
        return {"results": []}


@pytest.mark.parametrize(
    ("client", "expected_id", "expected_memory", "expected_hash"),
    [
        (_SearchClientFiltersOnly(), "123", "Prefers coffee", "abc"),
        (_SearchClientLegacyOnly(), "456", "Legacy coffee memory", "def"),
        (_SearchClientUnscopedFallback(), "654", "Unscoped coffee memory", None),
    ],
)
def test_search_memories_via_client_supports_current_and_legacy_signatures(client, expected_id, expected_memory, expected_hash):
    results = _search_memories_via_client(client, "coffee", "alice")
    assert results == [
        {
            "id": expected_id,
            "memory": expected_memory,
            "hash": expected_hash,
            "created_at": None,
            "updated_at": None,
            "score": 0.91 if expected_id == "123" else None,
        }
    ]


@pytest.mark.parametrize(
    ("client", "expected_id", "expected_memory"),
    [
        (_ListClientFiltersOnly(), "789", "Tea note"),
        (_ListClientLegacyOnly(), "987", "Legacy tea note"),
        (_ListClientUnscopedFallback(), "321", "Unscoped tea note"),
    ],
)
def test_list_memories_via_client_supports_current_and_legacy_signatures(client, expected_id, expected_memory):
    results = _list_memories_via_client(client, "alice")
    assert results == [
        {
            "id": expected_id,
            "memory": expected_memory,
            "hash": None,
            "created_at": None,
            "updated_at": None,
            "score": None,
        }
    ]


def test_score_memory_match_prefers_direct_query_match():
    assert _score_memory_match("backend smoke redis memory on Unraid with Postgres", "redis memory") == 1.0
    assert _score_memory_match("backend smoke redis memory on Unraid with Postgres", "redis unraid") == 1.0
    assert _score_memory_match("backend smoke redis memory on Unraid with Postgres", "missing tokens") is None


def test_search_memories_from_list_uses_local_matching_when_search_is_unavailable():
    results = _search_memories_from_list(_SearchFallsBackToListClient(), "redis Unraid", "alice")

    assert results == [
        {
            "id": "1",
            "memory": "backend smoke redis memory on Unraid with Postgres",
            "hash": None,
            "created_at": None,
            "updated_at": None,
            "score": 1.0,
        }
    ]


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *args, **kwargs):
        return self

    def all(self):
        return list(self._rows)


class _FakeSearchDB:
    def __init__(self, rows):
        self._rows = rows
        self.added = []
        self.committed = False
        self.closed = False

    def query(self, _model):
        return _FakeQuery(self._rows)

    def add(self, item):
        self.added.append(item)

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_search_memory_falls_back_to_list_when_backend_search_returns_empty(monkeypatch):
    fake_db = _FakeSearchDB(
        [
            type(
                "MemoryRow",
                (),
                {
                    "id": "11111111-1111-1111-1111-111111111111",
                    "content": "backend smoke elasticsearch memory on Unraid with Postgres",
                    "content_hash": None,
                    "created_at": datetime.datetime(2026, 4, 18, 10, 0, 0),
                    "updated_at": datetime.datetime(2026, 4, 18, 10, 1, 0),
                },
            )()
        ]
    )
    fake_user = type("User", (), {"id": "alice-id"})()
    fake_app = type("App", (), {"id": "app-id"})()

    monkeypatch.setattr("app.mcp_server.get_memory_client_safe", lambda: _SearchReturnsNoResultsClient())
    monkeypatch.setattr("app.mcp_server.SessionLocal", lambda: fake_db)
    monkeypatch.setattr("app.mcp_server.get_user_and_app", lambda db, user_id, app_id: (fake_user, fake_app))
    monkeypatch.setattr("app.mcp_server.check_memory_access_permissions", lambda db, memory, app_id: True)
    monkeypatch.setattr("app.mcp_server.MemoryAccessLog", lambda **kwargs: kwargs)

    user_token = user_id_var.set("alice")
    client_token = client_name_var.set("openmemory")
    try:
        result = await search_memory("elasticsearch memory")
    finally:
        user_id_var.reset(user_token)
        client_name_var.reset(client_token)

    assert "backend smoke elasticsearch memory on Unraid with Postgres" in result
    assert "2026-04-18T10:00:00" in result
    assert fake_db.committed is True
    assert fake_db.closed is True


@pytest.mark.asyncio
async def test_list_memories_falls_back_to_database_when_client_returns_empty(monkeypatch):
    fake_db = _FakeSearchDB(
        [
            type(
                "MemoryRow",
                (),
                {
                    "id": "11111111-1111-1111-1111-111111111111",
                    "content": "backend smoke elasticsearch memory on Unraid with Postgres",
                    "content_hash": None,
                    "created_at": datetime.datetime(2026, 4, 18, 10, 0, 0),
                    "updated_at": datetime.datetime(2026, 4, 18, 10, 1, 0),
                },
            )()
        ]
    )
    fake_user = type("User", (), {"id": "alice-id"})()
    fake_app = type("App", (), {"id": "app-id"})()

    monkeypatch.setattr("app.mcp_server.get_memory_client_safe", lambda: _SearchReturnsNoResultsClient())
    monkeypatch.setattr("app.mcp_server.SessionLocal", lambda: fake_db)
    monkeypatch.setattr("app.mcp_server.get_user_and_app", lambda db, user_id, app_id: (fake_user, fake_app))
    monkeypatch.setattr("app.mcp_server.check_memory_access_permissions", lambda db, memory, app_id: True)
    monkeypatch.setattr("app.mcp_server.MemoryAccessLog", lambda **kwargs: kwargs)

    user_token = user_id_var.set("alice")
    client_token = client_name_var.set("openmemory")
    try:
        result = await list_memories()
    finally:
        user_id_var.reset(user_token)
        client_name_var.reset(client_token)

    assert "backend smoke elasticsearch memory on Unraid with Postgres" in result
    assert "2026-04-18T10:00:00" in result
    assert fake_db.committed is True
    assert fake_db.closed is True


class _FakeDB:
    pass


@pytest.mark.asyncio
async def test_update_configuration_persists_and_returns_explicit_nulls(monkeypatch):
    current_config = {
        "openmemory": {"custom_instructions": "old"},
        "mem0": {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "api_key": "env:OPENAI_API_KEY",
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "api_key": "env:OPENAI_API_KEY",
                },
            },
            "vector_store": {"provider": "qdrant", "config": {"host": "mem0_store", "port": 6333}},
        },
    }
    saved = {}
    reset_calls = []

    monkeypatch.setattr("app.routers.config.get_config_from_db", lambda db: current_config.copy())
    monkeypatch.setattr("app.routers.config.save_config_to_db", lambda db, config: saved.setdefault("config", config))
    monkeypatch.setattr("app.routers.config.reset_memory_client", lambda: reset_calls.append(True))

    payload = ConfigSchema(
        openmemory=OpenMemoryConfig(custom_instructions=None),
        mem0=Mem0Config(
            llm=LLMProvider(
                provider="ollama",
                config=LLMConfig(
                    model="mistral:7b",
                    temperature=0.1,
                    max_tokens=2000,
                    api_key="env:OPENAI_API_KEY",
                    ollama_base_url="",
                ),
            ),
            embedder=EmbedderProvider(
                provider="ollama",
                config=EmbedderConfig(
                    model="nomic-embed-text",
                    api_key="env:OPENAI_API_KEY",
                    ollama_base_url="",
                ),
            ),
            vector_store=None,
        ),
    )

    response = await update_configuration(payload, db=_FakeDB())

    assert response == saved["config"]
    assert response["openmemory"]["custom_instructions"] is None
    assert response["mem0"]["vector_store"] is None
    assert response["mem0"]["llm"]["provider"] == "ollama"
    assert len(reset_calls) == 1


@pytest.mark.asyncio
async def test_update_configuration_preserves_openai_compatible_base_urls(monkeypatch):
    current_config = get_default_configuration()
    saved = {}

    monkeypatch.setattr("app.routers.config.get_config_from_db", lambda db: current_config.copy())
    monkeypatch.setattr("app.routers.config.save_config_to_db", lambda db, config: saved.setdefault("config", config))
    monkeypatch.setattr("app.routers.config.reset_memory_client", lambda: None)

    payload = ConfigSchema(
        openmemory=OpenMemoryConfig(custom_instructions=None),
        mem0=Mem0Config(
            llm=LLMProvider(
                provider="openai",
                config=LLMConfig(
                    model="tinyllama:latest",
                    temperature=0.1,
                    max_tokens=2000,
                    api_key="env:API_KEY",
                    openai_base_url="http://ollama:11434/v1",
                    ollama_base_url=None,
                ),
            ),
            embedder=EmbedderProvider(
                provider="openai",
                config=EmbedderConfig(
                    model="nomic-embed-text:latest",
                    api_key="env:API_KEY",
                    openai_base_url="http://ollama:11434/v1",
                    ollama_base_url=None,
                ),
            ),
            vector_store=None,
        ),
    )

    response = await update_configuration(payload, db=_FakeDB())

    assert response["mem0"]["llm"]["config"]["api_key"] == "env:API_KEY"
    assert response["mem0"]["llm"]["config"]["openai_base_url"] == "http://ollama:11434/v1"
    assert response["mem0"]["embedder"]["config"]["api_key"] == "env:API_KEY"
    assert response["mem0"]["embedder"]["config"]["openai_base_url"] == "http://ollama:11434/v1"
    assert "ollama_base_url" not in response["mem0"]["llm"]["config"] or response["mem0"]["llm"]["config"]["ollama_base_url"] is None


def test_get_default_configuration_prefers_ollama_when_base_url_is_set(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDER_PROVIDER", raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    monkeypatch.setenv("LLM_MODEL", "mistral:7b")
    monkeypatch.setenv("EMBEDDER_MODEL", "nomic-embed-text")

    config = get_default_configuration()

    assert config["mem0"]["llm"]["provider"] == "ollama"
    assert config["mem0"]["llm"]["config"]["ollama_base_url"] == "http://host.docker.internal:11434"
    assert config["mem0"]["llm"]["config"]["model"] == "mistral:7b"
    assert config["mem0"]["embedder"]["provider"] == "ollama"
    assert config["mem0"]["embedder"]["config"]["ollama_base_url"] == "http://host.docker.internal:11434"
    assert config["mem0"]["embedder"]["config"]["model"] == "nomic-embed-text"


def test_get_default_configuration_treats_auto_provider_as_unset(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    monkeypatch.setenv("EMBEDDER_PROVIDER", "auto")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    monkeypatch.setenv("LLM_MODEL", "mistral:7b")
    monkeypatch.setenv("EMBEDDER_MODEL", "nomic-embed-text")

    config = get_default_configuration()

    assert config["mem0"]["llm"]["provider"] == "ollama"
    assert config["mem0"]["embedder"]["provider"] == "ollama"


def test_get_default_memory_config_builds_elasticsearch_ssl_config(monkeypatch):
    monkeypatch.setenv("ELASTICSEARCH_HOST", "elasticsearch")
    monkeypatch.setenv("ELASTICSEARCH_PORT", "9200")
    monkeypatch.setenv("ELASTICSEARCH_USER", "elastic")
    monkeypatch.setenv("ELASTICSEARCH_PASSWORD", "changeme")
    monkeypatch.setenv("ELASTICSEARCH_USE_SSL", "true")
    monkeypatch.setenv("ELASTICSEARCH_VERIFY_CERTS", "false")

    config = get_default_memory_config()

    assert config["vector_store"]["provider"] == "elasticsearch"
    assert config["vector_store"]["config"]["host"] == "https://elasticsearch"
    assert config["vector_store"]["config"]["port"] == 9200
    assert config["vector_store"]["config"]["user"] == "elastic"
    assert config["vector_store"]["config"]["password"] == "changeme"
    assert config["vector_store"]["config"]["use_ssl"] is True
    assert config["vector_store"]["config"]["verify_certs"] is False


def test_get_default_memory_config_builds_opensearch_auth_and_ssl_config(monkeypatch):
    monkeypatch.delenv("ELASTICSEARCH_HOST", raising=False)
    monkeypatch.delenv("ELASTICSEARCH_PORT", raising=False)
    monkeypatch.setenv("OPENSEARCH_HOST", "opensearch")
    monkeypatch.setenv("OPENSEARCH_PORT", "9200")
    monkeypatch.setenv("OPENSEARCH_USER", "admin")
    monkeypatch.setenv("OPENSEARCH_PASSWORD", "adminadmin")
    monkeypatch.setenv("OPENSEARCH_USE_SSL", "true")
    monkeypatch.setenv("OPENSEARCH_VERIFY_CERTS", "false")

    config = get_default_memory_config()

    assert config["vector_store"]["provider"] == "opensearch"
    assert config["vector_store"]["config"]["host"] == "opensearch"
    assert config["vector_store"]["config"]["port"] == 9200
    assert config["vector_store"]["config"]["user"] == "admin"
    assert config["vector_store"]["config"]["password"] == "adminadmin"
    assert config["vector_store"]["config"]["use_ssl"] is True
    assert config["vector_store"]["config"]["verify_certs"] is False


def test_ensure_vector_store_dimensions_uses_ollama_probe(monkeypatch):
    monkeypatch.delenv("EMBEDDER_DIMENSIONS", raising=False)
    monkeypatch.setattr("app.utils.memory._probe_ollama_embedding_dimensions", lambda model, base_url: 768)

    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {"collection_name": "openmemory", "host": "mem0_store", "port": 6333},
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text:latest",
                "ollama_base_url": "http://ollama:11434",
            },
        },
    }

    updated = _ensure_vector_store_dimensions(config)

    assert updated["vector_store"]["config"]["embedding_model_dims"] == 768


def test_ensure_vector_store_dimensions_uses_openai_compatible_probe(monkeypatch):
    monkeypatch.delenv("EMBEDDER_DIMENSIONS", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    monkeypatch.setattr(
        "app.utils.memory._probe_openai_compatible_embedding_dimensions",
        lambda model, base_url, api_key: 768,
    )

    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {"collection_name": "openmemory", "host": "mem0_store", "port": 6333},
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "nomic-embed-text:latest",
                "api_key": "env:OPENAI_API_KEY",
                "openai_base_url": "http://ollama:11434/v1",
            },
        },
    }

    updated = _ensure_vector_store_dimensions(config)

    assert updated["vector_store"]["config"]["embedding_model_dims"] == 768


def test_repair_qdrant_collection_deletes_empty_mismatched_collection(monkeypatch):
    deleted = []

    class _CollectionInfo:
        class config:
            class params:
                class vectors:
                    size = 1536

    class _FakeQdrantClient:
        def __init__(self, **kwargs):
            assert kwargs["host"] == "mem0_store"
            assert kwargs["port"] == 6333

        def collection_exists(self, collection_name):
            assert collection_name == "openmemory"
            return True

        def get_collection(self, collection_name):
            assert collection_name == "openmemory"
            return _CollectionInfo()

        def delete_collection(self, collection_name):
            deleted.append(collection_name)

    monkeypatch.setattr("app.utils.memory.QdrantClient", _FakeQdrantClient)
    monkeypatch.setattr("app.utils.memory._count_active_memories", lambda: 0)

    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "openmemory",
                "host": "mem0_store",
                "port": 6333,
                "embedding_model_dims": 768,
            },
        }
    }

    repaired = _repair_qdrant_collection_if_needed(config)

    assert repaired is config
    assert deleted == ["openmemory"]


def test_get_default_memory_config_prefers_faiss_over_default_qdrant(monkeypatch):
    monkeypatch.setenv("FAISS_PATH", "/mem0/storage/faiss")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("LLM_MODEL", "tinyllama:latest")
    monkeypatch.setenv("EMBEDDER_MODEL", "nomic-embed-text")

    config = get_default_memory_config()

    assert config["vector_store"]["provider"] == "faiss"
    assert config["vector_store"]["config"]["path"] == "/mem0/storage/faiss"


def test_get_default_memory_config_treats_auto_provider_as_unset(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    monkeypatch.setenv("EMBEDDER_PROVIDER", "auto")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("LLM_MODEL", "tinyllama:latest")
    monkeypatch.setenv("EMBEDDER_MODEL", "nomic-embed-text")

    config = get_default_memory_config()

    assert config["llm"]["provider"] == "ollama"
    assert config["embedder"]["provider"] == "ollama"


def test_drop_none_entries_removes_provider_incompatible_nulls():
    value = {
        "provider": "openai",
        "config": {
            "model": "tinyllama:latest",
            "api_key": "env:API_KEY",
            "openai_base_url": "http://ollama:11434/v1",
            "ollama_base_url": None,
        },
        "vector_store": None,
    }

    cleaned = _drop_none_entries(value)

    assert cleaned == {
        "provider": "openai",
        "config": {
            "model": "tinyllama:latest",
            "api_key": "env:API_KEY",
            "openai_base_url": "http://ollama:11434/v1",
        },
    }
