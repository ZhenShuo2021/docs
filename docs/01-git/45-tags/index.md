---
title: 幫 Git 上標籤
author: zsl0621
description: 幫 Git 上標籤
tags:
  - Git
  - Programming
keywords:
  - Git
  - Programming
last_update:
  date: 2024-10-20T02:21:33+08:00
  author: zsl0621
---


# 幫 Git 上標籤

用標籤標示重要版本，分為兩種lightweight 和 annotated，官方建議使用 annotated。

## 新增標籤

- 新增 Annotated 標籤
  ```bash
  git tag -a 'v0.1.0' -m 'commend messages'
  ```

- 查看標籤
  ```bash
  git tag -n
  ```

- 為先前 commit 加標籤
  ```bash
  git tag -a '1.0.dev' 3b7de7f
  ```

- **推送標籤**
  ```bash
  git push origin my-annotated-tag
  ```

- 推送所有標籤
  ```bash
  git push --tags
  ```

## 刪除標籤

- 刪除本地標籤
  ```bash
  git tag -d <tag_name>
  ```

- 刪除遠端標籤
  ```bash
  git push <upstream> :refs/tags/<tag_name>
  ```
