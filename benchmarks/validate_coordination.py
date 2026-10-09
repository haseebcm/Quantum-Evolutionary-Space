"""Network-isolated worker coordination acceptance against actual PostgreSQL."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from validate_remote_stores import available_port, docker, ready

from qes.postgres_queue import PostgresTaskQueue
from qes.task_queue import StaleLeaseError


def wait_for(predicate, timeout=60):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if predicate():
            return
        time.sleep(.03)
    raise AssertionError('coordination scenario timed out')


def validate(image: str) -> dict:
    name='qes-coordination-'+uuid4().hex[:10]
    network=name+'-net'
    database=name+'-db'
    volume=name+'-data'
    password=uuid4().hex
    port=available_port()
    host=f'postgresql://postgres:{password}@127.0.0.1:{port}/postgres'
    remote=f'postgresql://postgres:{password}@{database}:5432/postgres'
    workers=[]
    queue=None
    docker('network','create',network)
    docker('volume','create',volume)
    try:
        docker('run','-d','--name',database,'--network',network,'-p',f'127.0.0.1:{port}:5432',
               '-v',volume+':/var/lib/postgresql/data','-e','POSTGRES_PASSWORD='+password,'postgres:16-alpine')
        ready('postgres',host)
        queue=PostgresTaskQueue(host)
        report={'postgres':'16','worker_containers':2}
        ids=[queue.submit('qes.search',{'max_evaluations':60,'seed':i},idempotency_key=f'load-{i}') for i in range(30)]
        for i in range(2):
            worker=name+f'-worker-{i}'
            workers.append(worker)
            docker('run','-d','--name',worker,'--network',network,'--cpus','1','--memory','256m',
                   '--read-only','--tmpfs','/tmp','-e','QES_POSTGRES_DSN='+remote,image,'--interval','.05')
        wait_for(lambda:queue.operational_snapshot()['counts']['completed']==30)
        assert all(queue.get(task_id)['attempts']==1 for task_id in ids)
        assert queue.operational_snapshot()['live_workers']==2
        assert queue.get(ids[0],tenant_id='other') is None
        for worker in workers:
            docker('exec',worker,'qes-status','--require-worker')
            docker('stop','--time','10',worker)
            assert docker('inspect','--format','{{.State.ExitCode}}',worker)=='0'
        report['network_worker_load_and_drain']='passed'
        assert queue.submit('qes.search',{'max_evaluations':60,'seed':0},idempotency_key='load-0')==ids[0]
        report['durable_idempotent_submission']='passed'
        try:
            queue.submit('qes.search',{},idempotency_key='load-0')
        except ValueError:
            pass
        else:
            raise AssertionError('conflicting idempotency content accepted')

        task=queue.submit('qes.search',{})
        partitioned=name+'-partitioned'
        workers.append(partitioned)
        code="""
import os,time
from qes import PostgresTaskQueue,StaleLeaseError
q=PostgresTaskQueue(os.environ['QES_POSTGRES_DSN'])
lease=q.claim('partitioned',lease_seconds=2)
assert lease is not None
print('claimed',flush=True)
time.sleep(6)
q.close()
q=PostgresTaskQueue(os.environ['QES_POSTGRES_DSN'])
try:
    q.complete(lease,{'obsolete':True})
except StaleLeaseError:
    print('fenced',flush=True)
else:
    raise AssertionError('stale writer accepted')
q.close()
"""
        docker('run','-d','--name',partitioned,'--network',network,'-e','QES_POSTGRES_DSN='+remote,
               '--entrypoint','python',image,'-c',code)
        wait_for(lambda:queue.get(task)['status']=='running')
        docker('network','disconnect',network,partitioned)
        wait_for(lambda:queue.operational_snapshot()['expired_leases']==1)
        lease=queue.claim('replacement',lease_seconds=30)
        assert lease.task_id==task and lease.attempt==2
        queue.complete(lease,{'fresh':True})
        docker('network','connect',network,partitioned)
        wait_for(lambda:docker('inspect','--format','{{.State.Running}}',partitioned)=='false')
        assert docker('inspect','--format','{{.State.ExitCode}}',partitioned)=='0',docker('logs',partitioned)
        assert 'fenced' in docker('logs',partitioned)
        assert queue.get(task)['result']=={'fresh':True}
        report['partition_rejoin_stale_worker_fencing']='passed'

        queue.submit('qes.search',{})
        with patch('time.time',return_value=-1000000000):
            skewed=queue.claim('skewed-clock',lease_seconds=2)
        assert 0<skewed.expires_at-time.time()<3
        queue.renew(skewed,lease_seconds=30)
        try:
            queue.complete(replace(skewed,tenant_id='other'),{})
        except StaleLeaseError:
            pass
        else:
            raise AssertionError('wrong tenant lease accepted')
        queue.complete(skewed,'scalar result')
        assert queue.get(skewed.task_id)['result']=='scalar result'
        report['database_clock_renewal_and_tenant_fencing']='passed'

        queue.submit('qes.search',{})
        delayed=queue.claim('delayed',lease_seconds=1)
        blocker=queue._driver.connect(host)
        try:
            with blocker.cursor() as cursor:
                cursor.execute('SELECT task_id FROM qes_queue_tasks WHERE task_id=%s FOR UPDATE',(delayed.task_id,))
            with ThreadPoolExecutor(1) as pool:
                future=pool.submit(queue.complete,delayed,{'too_late':True})
                time.sleep(1.2)
                blocker.rollback()
                try:
                    future.result(timeout=10)
                except StaleLeaseError:
                    pass
                else:
                    raise AssertionError('completion accepted after lease expired during lock wait')
        finally:
            blocker.close()
        recovered=queue.claim('replacement')
        assert recovered.task_id==delayed.task_id
        queue.complete(recovered,{'fresh_after_lock_wait':True})
        report['lock_wait_expiry_fencing']='passed'


        small=PostgresTaskQueue(host,namespace='capacity_test',max_records=1)
        def submit(i):
            client=PostgresTaskQueue(host,namespace='capacity_test',max_records=1)
            try:
                return client.submit('work',i)
            except OverflowError:
                return None
            finally:
                client.close()
        try:
            with ThreadPoolExecutor(4) as pool:
                result=list(pool.map(submit,range(4)))
            assert sum(value is not None for value in result)==1
            capped=small.claim('worker')
            small.fail(capped,'permanent',retry=False)
            assert small.purge_terminal(before=time.time()+1)==1
        finally:
            small.close()
        report['atomic_capacity_and_retention']='passed'
        retry_id=queue.submit('qes.search',{},max_attempts=1)
        assert queue.claim('lost',lease_seconds=.2).task_id == retry_id
        wait_for(lambda:queue.operational_snapshot()['expired_leases']==1)
        assert queue.claim('replacement') is None
        assert queue.get(retry_id)['status']=='failed'
        report['attempt_exhaustion']='passed'
        docker('kill',database)
        try:
            queue.claim('offline')
        except Exception:
            pass
        else:
            raise AssertionError('offline database accepted claim')
        docker('start',database)
        ready('postgres',host)
        assert queue.get(task)['result']=={'fresh':True}
        report['database_restart_durability_and_reconnect']='passed'
        docker('exec',database,'pg_dump','-U','postgres','-d','postgres','--file','/tmp/queue-backup.sql')
        docker('exec',database,'psql','-U','postgres','-d','postgres','-c',
               'DROP TABLE qes_queue_tasks,qes_queue_workers,qes_queue_meta,capacity_test_tasks,capacity_test_workers,capacity_test_meta,qes_test')
        docker('exec',database,'psql','-v','ON_ERROR_STOP=1','-U','postgres','-d','postgres','-f','/tmp/queue-backup.sql')
        assert queue.get(task)['result']=={'fresh':True}
        report['full_queue_backup_restore']='passed'
        report['final_metrics']=queue.operational_snapshot()
        return report
    finally:
        if queue:
            queue.close()
        for worker in workers:
            subprocess.run(['docker','rm','-f',worker],capture_output=True,timeout=30)
        subprocess.run(['docker','rm','-f',database],capture_output=True,timeout=30)
        subprocess.run(['docker','volume','rm',volume],capture_output=True,timeout=30)
        subprocess.run(['docker','network','rm',network],capture_output=True,timeout=30)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--docker-image',required=True)
    parser.add_argument('--json',required=True)
    args=parser.parse_args()
    report=validate(args.docker_image)
    Path(args.json).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
