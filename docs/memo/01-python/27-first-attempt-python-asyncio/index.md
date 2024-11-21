---
title: 初嘗非同步操作 asyncio
description: 初嘗 Python 非同步操作 asyncio
tags:
  - Programming
  - Python
keywords:
  - Programming
  - Python
  - asyncio
last_update:
  date: 2024-11-22T02:42:30+08:00
  author: zsl0621
---

# 初嘗非同步操作 asyncio

第一次寫 Python 非同步語法的紀錄，網路上很多非同步文章，但是沒看過有把非同步放在獨立線程中執行的程式碼實例，於是自己寫了一個。

## 程式碼說明
首先先建立一個 dataclass 用於把要執行函式以及函式輸入打包

```py
@dataclass
class Task:
    task_id: str
    func: Callable[..., Any]
    args: Tuple[Any, ...] = ()
    kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.kwargs = self.kwargs or {}
```

接下來就是建立負責管理事件迴圈和線程的類別，首先初始化如下，項目有點多。

- max_workers 用於限制最高並發數量
- is_running 是程式旗標，標注線程是否還在運行
- task_queue 用於緩衝任務，因為有 max_workers 限制最高並發數量
- sem 是限制最高並發的 semaphore 鎖

```py
class AsyncService:
    def __init__(self, logger: Logger, max_workers: int = 5) -> None:
        self.max_workers = max_workers
        self.logger = logger

        self.is_running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.sem = asyncio.Semaphore(self.max_workers)
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self.task_queue: queue.Queue[Task] = queue.Queue()
        self.results: Dict[str, Any] = {}
        self.current_tasks: list[asyncio.Task[Any]] = []
```

接下來從程式碼執行的部分開始說明會比較清楚，要執行時 `Task` dataclass 經過一連串傳遞最後會被丟進 `_run_task` 這個方法中，先 await 結果，最後用線程鎖控制輸出寫入。雖然只有單線程應該不需要線程鎖，但是考量三個原因還是把他加上去：

1. 要是哪天忘了他只能用單線程，把程式改成多線程那就造成競爭危害了，防範於未然
2. 筆者很弱，雖然 99.9999% 確定不會造成競爭危害（因為事件迴圈本質是順序執行，除非在寫入時也 await 才有可能造成非原子操作導致競爭危害），但是不想賭那個 0.00001% 的問題
3. 相較 io 任務而言這個鎖的開銷簡直微乎其微

```py
async def _run_task(self, task: Task) -> Any:
    async with self.sem:
        print(
            f"Task {task.func.__name__} with args {task.args} and kwargs {task.kwargs} start running!"
        )
        try:
            result = await task.func(*task.args, **task.kwargs)
            with self._lock:
                self.results[task.task_id] = result
            return result
        except Exception as e:
            self.logger.error(f"Error processing task {task.task_id}: {e}")
            with self._lock:
                self.results[task.task_id] = None
```

接下來我們繼續介紹程式架構，`_run_task` 被包裝在 `_process_task` 中，前者用於真正執行任務，後者用於任務管理，是一個中間人的角色，負責從 task_queue 中取出任務註冊到事件迴圈中，並且放到 current_tasks 這個列表中準備執行。命名這麼相似的原因是在我的專案中這兩個都是繼承自一個抽象類別，後者名稱已經由父類決定好是 `_process_task`，那真正執行只好取 run 了，好啦其實是自己沒創意。

```py
async def _process_tasks(self) -> None:
    while True:
        self.current_tasks = [task for task in self.current_tasks if not task.done()]

        if self.task_queue.empty() and not self.current_tasks:
            break

        while not self.task_queue.empty() and len(self.current_tasks) < self.max_workers:
            try:
                task = self.task_queue.get_nowait()
                task_obj = asyncio.create_task(self._run_task(task))
                self.current_tasks.append(task_obj)
            except queue.Empty:
                break

        if self.current_tasks:
            await asyncio.wait(self.current_tasks, return_when=asyncio.FIRST_COMPLETED)
```

以上都還是私有函數不會被直接呼叫，而且只包含事件註冊和運行事件，還不包含運行事件迴圈以及把事件迴圈放到線程中執行（文章開頭目的：把非同步放在獨立線程中執行的程式碼實例），所以又要包裝在 `_start_event_loop` 還有 `_check_thread`，加上這兩個方法這樣整個工具函式就完成九成了，剩下一成是將任務放入佇列和停止線程跟事件迴圈，這裡我以 `add_task` 和 `add_task` 作為外部接口可以放入單一任務或是任務列表，以及 `stop` 方法作為結束所有任務的阻塞訊號。完整程式碼如下，也可以在[我的 Github 中找到](https://github.com/ZhenShuo2021/blog-script/tree/main/asyncio)：


```py
import asyncio
import queue
import threading
import time
from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Any, Dict, Tuple, Callable, Optional

from help import BLOCK_MSG, NOT_BLOCK_MSG, io_task, print_thread_id, timer


@dataclass
class Task:
    """Unified task container for both threading and async services."""

    task_id: str
    func: Callable[..., Any]
    args: Tuple[Any, ...] = ()
    kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.kwargs = self.kwargs or {}


class AsyncService:
    def __init__(self, logger: Logger, max_workers: int = 5) -> None:
        self.max_workers = max_workers
        self.logger = logger
        self.is_running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self.task_queue: queue.Queue[Task] = queue.Queue()
        self.results: Dict[str, Any] = {}
        self.current_tasks: list[asyncio.Task[Any]] = []
        self.sem = asyncio.Semaphore(self.max_workers)

    def start(self) -> None:
        self._check_thread()

    def add_task(self, task: Task) -> None:
        self.task_queue.put(task)
        self._check_thread()

    def add_tasks(self, tasks: list[Task]) -> None:
        for task in tasks:
            self.task_queue.put(task)
        self._check_thread()

    def get_result(self, task_id: str) -> Optional[Any]:
        with self._lock:
            return self.results.pop(task_id, None)

    def get_results(self, max_results: int = 0) -> Dict[str, Any]:
        with self._lock:
            if max_results <= 0:
                results_to_return = self.results.copy()
                self.results.clear()
                return results_to_return

            keys = list(self.results.keys())[:max_results]
            return {key: self.results.pop(key) for key in keys}

    async def _process_tasks(self) -> None:
        while True:
            self.current_tasks = [task for task in self.current_tasks if not task.done()]

            if self.task_queue.empty() and not self.current_tasks:
                break

            while not self.task_queue.empty() and len(self.current_tasks) < self.max_workers:
                try:
                    task = self.task_queue.get_nowait()
                    task_obj = asyncio.create_task(self._run_task(task))
                    self.current_tasks.append(task_obj)
                except queue.Empty:
                    break

            if self.current_tasks:
                await asyncio.wait(self.current_tasks, return_when=asyncio.FIRST_COMPLETED)

    async def _run_task(self, task: Task) -> Any:
        async with self.sem:
            print(
                f"Task {task.func.__name__} with args {task.args} and kwargs {task.kwargs} start running!"
            )
            try:
                result = await task.func(*task.args, **task.kwargs)
                with self._lock:
                    self.results[task.task_id] = result
                return result
            except Exception as e:
                self.logger.error(f"Error processing task {task.task_id}: {e}")
                with self._lock:
                    self.results[task.task_id] = None

    def _check_thread(self) -> None:
        with self._lock:
            if not self.is_running or self.thread is None or not self.thread.is_alive():
                self.is_running = True
                self.thread = threading.Thread(target=self._start_event_loop)
                self.thread.start()

    def _start_event_loop(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._process_tasks())
        finally:
            self.loop.close()
            self.loop = None
            self.is_running = False
            self.current_tasks.clear()

    def stop(self, timeout: Optional[float] = None) -> None:
        if self.thread is not None:
            self.thread.join(timeout=timeout)
            print(f"\n===no job! clearing thread {self.thread.native_id}===")
            self.thread = None
            self.is_running = False
            print(f"===thread cleared! result: {self.thread}===\n")


@timer
def test() -> None:
    print_thread_id()
    logger = getLogger()
    task_groups = [
        [(1, "A1"), (2, "A2"), (3, "A3")],
        [(3, "B1"), (4, "B2"), (5, "B3")],
        [(3, "C1"), (4, "C2"), (5, "C3")],
        [(1, "D1"), (2, "D2"), (3, "D3")],
    ]

    manager = AsyncService(logger, max_workers=5)

    # 提交第一批任務，使用新的add_tasks方法
    for group in task_groups[:-1]:
        tasks = [Task(task[1], io_task, task) for task in group]
        manager.add_tasks(tasks)

    print(NOT_BLOCK_MSG)

    # 模擬主執行緒工作需要 2.5 秒，在程式中間取得結果
    # 會顯示 A1/A2 的結果，因為它們在 2.5 秒後完成
    time.sleep(2.5)
    results = manager.get_results()
    print(NOT_BLOCK_MSG, "(2s waiting for main thread itself)")  # not blocked
    for result in results:
        print(result)

    # 等待子執行緒結束，造成阻塞
    manager.stop()
    print(BLOCK_MSG)

    for _ in range(3):
        results = manager.get_results()
        for result in results:
            print(result)

    # 在thread關閉後提交第二批任務
    tasks = [Task(task[1], io_task, task) for task in task_groups[-1]]
    manager.add_tasks(tasks)
    manager.stop()
    results = manager.get_results()
    for result in results:
        print(result)


if __name__ == "__main__":
    print_thread_id()
    test()
```

這裡包含了一個簡單範例 `test`，實際運行是 11 秒和預期相符，讀者可以自行計算秒數驗證是否和理論相符。

## 自我檢討和心得
搜尋資料時看到有人建議撰寫主程式是非同步，然後把同步語法放到子線程中執行會比較好，寫的時候不太認同，真的用函式的時候就認同了，因為即使已經包裝成只要 `add_task` 和輸入 `Task` 就可以使用還是不太方便。

第二個是層層包裹的語句造成理解不易，使用 `add_task` 加入任務後會經過 `_check_thread` 確認線程是否存活並且建立線程，線程裡面要使用 `_start_event_loop` 建立事件迴圈，再使用 `_process_tasks` 把事件註冊到迴圈中，最後用 `_run_task` 把 `Task` dataclass 解包並且執行。這呼應第一個問題：如果去掉在子線程執行事件迴圈這件事，就可以刪掉前兩個方法，簡化為只需要註冊和運行而已，而且當主線程是非同步語法時可以省略 `_start_event_loop` 方法，因為不需要透過一個類別作為任務控制器管理事件迴圈和任務，自然也就不需要 `_start_event_loop`。會這樣寫的原因除了自己想練習以外，也是因為前一個練習是「把任務丟到子線程中執行」，所以依照同樣的想法寫了事件迴圈版本，結果比想像中的麻煩多了。不過都是試了才知道，畢竟網路上又沒這種文章，除非網路文章很爛否則筆者絕對不寫網路上已經存在的內容浪費讀者和自己彼此的時間。

第三是多個事件迴圈的管理運行，但筆者還沒到那個程度，所以 no comment。

第四有關效能，`task_queue`、`current_tasks` 和 `results` 疊床架屋，`task_queue` 用於暫存還沒執行的任務，`current_tasks` 存放已經從 `task_queue` 取出正在被 await 的任務，`results` 存放任務結果，例如每個 HTTP 請求的狀態碼，一個任務要三個物件管理有點浪費資源，但是為了實現「獨立線程中運行事件迴圈」好像也想不到什麼更好方法，有 semaphore 應該可以去掉 `task_queue` 和 `current_tasks`，不過 `results` 應該就沒辦法刪除，因為主線程可以不取資料，獨立線程又不知道到底能不能丟，只好全部暫存。附帶一提 `results` 不使用 queue 資料結構的原因是用戶可能會想根據 task_id 取得結果，但是 queue 只能從頭尾取值達不到這項要求。

最後補充，這個腳本跑 mypy --strict 可以過的唷。
