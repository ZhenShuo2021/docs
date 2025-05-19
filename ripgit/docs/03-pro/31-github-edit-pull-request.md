---
title: Github Pull Request
sidebar_label: Pull Request
slug: /github-pull-request
tags:
  - Git
  - Github
  - 教學
keywords:
  - Git
  - Github
  - 教學
last_update:
  date: 2025-05-12T18:32:30+08:00
  author: zsl0621
first_publish:
  date: 2025-05-12T18:32:30+08:00
---

# {{ $frontmatter.title }}

Pull request (PR) 不是 Git 的原生指令，是 Git 託管平台的加值服務，本身是分支的一種，目的是用於多人協作，把你的提交發給別人的 repo，又或者是避免在主分支提交也可以用 PR 工作流程。

## 怎麼發 Pull Request

1. fork repo, clone forked repo。
2. 修改程式碼，在子分支上提交後推送，流程依照 [團隊協作最佳實踐](../core/collaboration-best-practice)。
3. 到 Github 上，可以是原始 repo 或是 forked repo 都會顯示問你要不要發 PR，點進去修改內容完成。
4. 發送 PR 後，後續在該分支上的所有提交都會包含進該 PR 中。

## 編輯 Github 的 Pull Requests 合併衝突

如果你是 maintainer，遇到 PR 合併衝突應該如何解決呢？簡單的合併衝突可以在 Github WebUI 上直接修改，複雜的可以 clone 下來改：

```sh
# 使用 gh 指令比較方便
gh pr checkout <ID>
```

如果不想安裝 [gh cli](https://cli.github.com/) 也可以回歸原始 Git 指令完成

```sh
# 獲取 PR 內容
git fetch origin pull/<ID>/head:<BRANCH_NAME>

# 切換到 PR 分支
git switch <BRANCH_NAME>

# 推送並且建立新的 PR
git push origin BRANCH_NAME
```

這是官方最佳實踐，不用想東想西，官方就建議這樣做

- [Checking out pull requests locally](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/checking-out-pull-requests-locally)

## 合併兩個 PR

[跟負責合併的人說這個 PR 基於哪個 PR 就好了](https://github.com/orgs/community/discussions/22827)，實際就像是[這樣](https://github.com/nunocoracao/blowfish/pull/2059)。

## 多人共同編輯 PR

[不要這樣做](https://stackoverflow.com/questions/60556393/can-two-people-commit-code-to-the-same-pr-without-maintainer-privileges)，好的方式是拆成小問題發送多個 PR。
