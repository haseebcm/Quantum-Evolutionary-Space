"""Submit, execute, inspect and back up a bounded durable local task."""
from tempfile import TemporaryDirectory

from qes import RegisteredTaskWorker, SQLiteTaskQueue
from qes.worker import search_task

with TemporaryDirectory() as directory:
    queue = SQLiteTaskQueue(f"{directory}/tasks.sqlite")
    task_id = queue.submit("qes.search", {"lower": [-2, -2], "upper": [2, 2],
                                        "max_evaluations": 100, "seed": 7},
                           idempotency_key="example-search")
    worker = RegisteredTaskWorker(queue, {"qes.search": search_task}, owner="example")
    worker.run_one(lease_seconds=300)
    record = queue.get(task_id)
    assert record is not None and record["status"] == "completed"
    assert record["result"]["evaluations"] <= 100
    queue.backup(f"{directory}/backup.sqlite")
    print(record["result"])
    queue.close()
