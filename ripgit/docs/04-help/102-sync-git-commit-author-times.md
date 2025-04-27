---
title: 同步 Git 提交時間與作者資訊
sidebar_label: 同步提交時間和作者時間
slug: /sync-git-commit-author-times
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

# {{ $frontmatter.title }}

如何同步提交時間和作者時間？所謂的作者時間 (author_data) 是這個提交「被提交的當下」的時間，提交時間 (committer_data) 則是「被修改的提交時間」，其他的 committer_email, committer_name 也同理。

修改方式有兩種，如果只要修改提交時間的話 rebase 就可輕鬆完成，設定 N 是要修改的範圍：

```sh
git rebase HEAD~<N> --committer-date-is-author-date
```

如果要連 committer_name committer_email 都修改的話就需要請出 `git filter-repo`，一行指令搞定，以修改 `<branch-name>` 為例：

```sh
git filter-repo --force --commit-callback '
    commit.committer_name = commit.author_name
    commit.committer_email = commit.author_email
    commit.committer_date = commit.author_date
' --refs refs/heads/<branch-name>
```

這樣會把 committer_data 設定為 author_data，不使用 `--refs` 就會修改整個 repo。修改前請先把整個 repo 複製一份在新的那一份進行測試，不要直接修改原始 repo。
