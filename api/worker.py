from __future__ import annotations

from rq import Worker

from api.queue import get_redis


def main() -> None:
    redis_conn = get_redis()
    worker = Worker(["default"], connection=redis_conn)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()

