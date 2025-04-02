---
title: 遠端儲存庫設定
author: zsl0621
description: 上傳到遠端儲存庫的必要前置設定。
tags:
  - Git
  - 教學
keywords:
  - Git
  - 教學
last_update:
  date: 2024-09-10T06:07:33+08:00
  author: zsl0621
first_publish:
  date: 2024-09-10T06:07:33+08:00
---

# Git 遠端儲存庫設定

本文設定 SSH 連線和 GPG 簽名，選了兩篇照著做就對的文章，已經有優質文章就不需要重造輪子。

## 設定 SSH

Github 已不支援帳號密碼登入，只能用 SSH 認證。  

1. [產生ssh金鑰](https://www.maxlist.xyz/2022/12/22/github-ssh-setting/)
2. (選用) 避免使用私人電子郵件：隱藏信箱Setting>Email勾選 "Block command line pushes that expose my email"，這個設定會阻止你使用個人信箱推送，如果提交的 email 不是步驟三的隱藏信箱就會被阻止
3. (選用) 隱藏信箱，請到 `https://api.github.com/users/你的github名稱` 找到 id 欄位並且設定

```sh
git config --global user.email "{ID}+{你的github名稱}@users.noreply.github.com"
```

4. 測試 `ssh -T git@github.com`，出現 successfully authenticated 即成功，不用管 GitHub does not provide shell access 這句話。
5. (選用) 連接並且上傳既有的 github repo

```sh
git remote add origin git@github.com:your-username/your-repo.git
git push -u origin main
```

## 可選：設定 GPG 簽名

請直接看 [利用 GPG 簽署 git commit](https://blog.puckwang.com/posts/2019/sign_git_commit_with_gpg/) 的教學，圖文並茂且完整，並且包含常見錯誤的解決方法。  
如果要隱藏信箱在 GPG 設定時需使用剛剛設定的 noreply 信箱。  
如果已經有 GPG key，可以用以下指令刪除：

```sh
git config --global --unset-all user.signingkey
```
