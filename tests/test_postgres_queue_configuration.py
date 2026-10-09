import pytest

from qes.postgres_queue import PostgresTaskQueue


@pytest.mark.parametrize('kwargs', [{'namespace':'x; DROP TABLE jobs'}, {'namespace':'A'},
                                    {'max_records':0}, {'max_payload_bytes':True}])
def test_invalid_configuration_rejected_before_connecting(kwargs):
    with pytest.raises(ValueError):
        PostgresTaskQueue('unused', **kwargs)
