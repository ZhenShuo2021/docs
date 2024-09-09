---
title: '幫所有 commit 加上 GPG 簽名'
description: 幫 commit 加上已認證標籤。
tags:
  - Git
  - Programming
keywords:
  - Git
  - Programming
last_update:
  date: 2024-08-22T05:31:07+08:00
  author: zsl0621
---

# 幫所有過往 commit 加上 GPG 簽名

## 起因
發現 gpg 認證失敗沒辦法 commit，當時腦袋一熱就想著藉此機會把所有以前設定錯誤的金鑰全部移除，移除完再新增金鑰所有操作都正常除了以前所有的簽名資訊全部失敗以外...因為我把 Github 上面的金鑰也一起刪了導致舊 commit 在遠端上無法驗證。

不過現在想想可能是 gpg 金鑰過期了。

## 修復
很簡單，[一行指令完成](https://stackoverflow.com/questions/41882919/is-there-a-way-to-gpg-sign-all-previous-commits)

```sh
git filter-branch --commit-filter 'git commit-tree -S "$@";' -- --all
```

## 意外事件
但是遇到問題，出現以下錯誤
```sh
Rewrite COMMIT_ID (1/179) (0 seconds passed, remaining 0 predicted)    error: gpg failed to sign the data
could not write rewritten commit
```

gpg 無法簽名，不讓我重寫 commit，於是先檢查檢查 gpg 是否正常運作
```sh
gpg --list-secret-keys --keyid-format LONG
git config --global user.signingkey
echo "test" | gpg --batch --yes --clearsign
-----BEGIN PGP SIGNED MESSAGE-----
Hash: SHA256

test
gpg: 簽署時失敗: Inappropriate ioctl for device
gpg: [stdin]: clear-sign failed: Inappropriate ioctl for device
```

## 解決
發現是終端設定不知為何跑掉，使用以下指令告訴 gpg 使用當前終端操作，即可正常提交 commit
```sh
export GPG_TTY=$(tty)
```

最後可以用 `git log --show-signature` 檢查簽名情況。