---
title: 使用 git-filter-repo 批量修改 Git 提交訊息
sidebar_label: 批量修改提交訊息
slug: /batch-rename-commit-message
tags:
  - Git
  - 教學
keywords:
  - Git
  - 教學
last_update:
  date: 2025-04-06T16:52:07+08:00
  author: zsl0621
first_publish:
  date: 2025-04-06T16:52:07+08:00
---

如何批量修改提交訊息？這個功能一樣需要請出 `git filter-repo`

## 以修改 github-bot 的提交訊息為例

因為我設定他的提交格式是

```sh
[ci skip]chore: automated update at 2025-04-05T22:11:32+08:00
```

想要改成換行這樣比較好看，像是這種效果

```sh
chore: automated update at 2025-04-05T22:11:32+08:00

[ci skip]
```

使用 `git filter-repo` 腳本如下

```py
#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "git-filter-repo",
# ]
# ///

import re

from git_filter_repo import FilteringOptions, RepoFilter


def modify_commit_message(commit, metadata) -> None:
    pattern = r"^\[ci skip\]chore: automated update at ([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\+[0-9]{2}:[0-9]{2})$"
    message = commit.message.decode("utf-8")

    match = re.match(pattern, message)
    if match:
        datetime_str = match.group(1)
        new_message = f"chore: automated update at {datetime_str}\n\n[ci skip]"
        commit.message = new_message.encode("utf-8")


args = FilteringOptions.parse_args(["--force"])
repo_filter = RepoFilter(args, commit_callback=modify_commit_message)
repo_filter.run()
```

最後使用 `uv run script-name.py` 完成。

### 說明

這是 shebang 和 PEP 723 規範，使用 PEP 723 加上適當工具就可以直接執行自動下載依賴。

```py
#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "git-filter-repo",
# ]
# ///
```

modify_commit_message 是主要邏輯，regex 使用方式請見 [cheatsheet](/linux-command/regex)，教學請見 [RegexLearn](https://regexlearn.com/learn/regex101)。

```py
def modify_commit_message(commit, metadata) -> None:
    # 使用 regex 找到對應提交訊息
    pattern = r"^\[ci skip\]chore: automated update at ([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\+[0-9]{2}:[0-9]{2})$"
    message = commit.message.decode("utf-8")

    match = re.match(pattern, message)
    if match:
        # 符合的話就替換
        datetime_str = match.group(1)
        new_message = f"chore: automated update at {datetime_str}\n\n[ci skip]"
        commit.message = new_message.encode("utf-8")
```

## 以移除 cherry-pick message 為例，並且限制範圍

用了 `cherry-pick -x` 選項所有 commit 都多了一個註解

```txt
commit 643232ec6856ff40dddef6ebf1723a7da9f9498c
Author: ZhenShuo2021 <98386542+ZhenShuo2021@users.noreply.github.com>
Date:   Sat Apr 19 04:55:12 2025 +0800
    docs: update readme [ci skip]
    
    (cherry picked from commit 30da84e5b59f4a2719419c320308248375cacaa7)
```

現在反悔想要移除，使用指令完成，限制 refs feat 分支往前兩個提交：

```sh
git filter-repo --message-callback '
import re
try:
    message_str = message.decode("utf-8")
except UnicodeDecodeError:
    # Fall back to latin-1 or other encodings if utf-8 fails
    message_str = message.decode("latin-1")
    
modified_message = re.sub(r"\n\(cherry picked from commit [0-9a-f]+\)\n?", "\n", message_str)
return modified_message.encode("utf-8")
' --refs feat~2..feat --force
```
