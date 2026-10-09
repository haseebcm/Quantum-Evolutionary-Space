import math

import pytest

from qes.runtime import PostgresRuntimeStore, RedisRuntimeStore


class BytesRedis:
    def __init__(self):
        self.values = {}
        self.closed = False

    def hset(self, prefix, key, value):
        self.values[key.encode()] = value.encode()

    def hgetall(self, prefix):
        return self.values.copy()

    def close(self):
        self.closed = True


def test_redis_bytes_snapshot_finite_json_and_closed_lifecycle():
    client = BytesRedis()
    store = RedisRuntimeStore(client=client)
    store.put("job", {"value": 1})
    assert store.snapshot() == {"job": {"value": 1}}
    with pytest.raises(ValueError):
        store.put("invalid", {"value": math.nan})
    assert b"invalid" not in client.values
    store.close()
    store.close()
    assert client.closed
    with pytest.raises(RuntimeError, match="closed"):
        store.snapshot()


@pytest.mark.parametrize("kwargs", [{"key_prefix": ""}, {"socket_timeout": 0}, {"socket_timeout": math.inf}])
def test_redis_rejects_invalid_connection_configuration(kwargs):
    with pytest.raises(ValueError):
        RedisRuntimeStore(client=BytesRedis(), **kwargs)


def test_postgres_failed_rollback_preserves_original_error():
    class BrokenConnection:
        def cursor(self):
            raise OSError("original connection failure")

        def rollback(self):
            raise RuntimeError("rollback also failed")

    with pytest.raises(OSError, match="original connection failure"):
        PostgresRuntimeStore(connection=BrokenConnection())
