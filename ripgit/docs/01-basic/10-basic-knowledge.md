---
title: Git 基礎知識
slug: /basic-knowledge
tags:
  - Git
  - 教學
keywords:
  - Git
  - 教學
last_update:
  date: 2024-09-13T04:00:33+08:00
  author: zsl0621
first_publish:
  date: 2024-09-13T04:00:33+08:00
---

# {{ $frontmatter.title }}

本文介紹大的 picture 讓你可以宏觀的理解 Git 運作方式，先知道他在做什麼再學會怎麼操作他，網路很多教學文章都本末倒置，導致初學者指令貼上可以用，但是就是不清楚自己在做甚麼。

## Quick Overview

Git 是版本管理工具，紀錄版本的**單位**是一個一個的提交 (commit)，每個提交都會計算**獨一無二**的 hash 幫提交設定序號，並且**指向上次的提交**，這樣連起來就可以記錄版本歷史。除了基本的提交歷史順序，還有一個重要的功能是分支 (branch)，意思是從提交中分出支線，我們可以在支線進行功能開發、修復 bug 等工作，讓不同的開發作業可以獨立工作，最後在開發完成後將支線整合回主分支，這種機制使得不同開發人員能同時進行多項開發任務，而不會影響**主要分支的穩定性**。

<br/>

:::tip

1. 避免誤導所以不說合併 (merge) 而說整合，因為整合有各種方式達成不限於 `git merge`。
2. 提交 (commit) 同時是名詞也是動詞，把一個提交（名詞）提交到（動詞）儲存庫，閱讀時要稍微注意一下是名詞還是動詞。

:::

<br/>

上網查 Git 時一定都看過這種流程圖，以本圖為例，從主分支切出 login 分支，OTP 分支又基於 login 分支進行開發，確認功能穩定可行後再整合回主分支，這樣的意義在於不影響穩定的主分支，即便在某個分支開發過程中出現問題，也不會影響到其他開發者的進度或整體系統的運行。

```mermaid
gitGraph
  commit id: "..."
  checkout main
  commit id: "main A"
  branch login
  commit id: "login A"
  commit id: "login B"


  checkout login
  branch OTP
  commit id: "OTP A"

  checkout login
  commit id: "login C - fix bugs"

  checkout OTP
  cherry-pick id: "login C - fix bugs"
  commit id: "OTP B"

  checkout main
  merge login id: "merge login branch"

  checkout login
  commit id: "login D"

  checkout main
  merge OTP id: "merge OTP branch"
```

## 本地與遠端儲存庫的關係

> 此段落修改自[官方說明](https://git-scm.com/book/zh-tw/v2/%E9%96%8B%E5%A7%8B-Git-%E5%9F%BA%E7%A4%8E%E8%A6%81%E9%BB%9E)：三種狀態

前一個段落我們介紹完 Git 的宏觀用法，接下來我們來介紹他是如何運作的。Git 實際運作可以分為三個層面理解，分別是硬碟、本地儲存庫 (專案的 .git 資料夾)、遠端儲存庫 (Github/Gitlab)。你的硬碟不知道任何版本資訊，只負責儲存檔案當前狀態；本地儲存庫是專案目錄中的 .git 資料夾，此資料夾會紀錄所有版本資訊；遠端儲存庫則是最後上傳共享提交歷史的地方。撰寫程式時，每次想要紀錄版本就**提交**一個版本到本地儲存庫，完成一個段落後使用**推送**功能，將提交推送到遠端儲存庫，和其他成員進行協作開發。

:::info DVCS: 分散式版本控制

這個段落描述的版本控制系統架構描述了每個開發人員都有完全相同的提交歷史鏡像，這種架構稱為分散式版本控制系統。除此之外，另外 Git 是快照系統而不是差異系統。

不是很重要的知識，看過就好。

:::

## Git 的檔案狀態{#file-status}

也就是說真正負責管理版本歷史的只有本地端的儲存庫，遠端只是負責同步的鏡像版本，所以接下來我們說明本地儲存庫是如何管理檔案的。Git 會把檔案標記為三種狀態，分別是已提交（committed）、已預存（staged）和已修改（modified）：

1. **已提交**: 檔案己安全地存在你的本地儲存庫
2. **已預存**: 已將修改的檔案新增至索引，準備提交至儲存庫（準備被提交但是尚未提交）
3. **己修改**: 檔案已被更改，但尚未加入至索引（該檔案的修改沒有被儲存庫追蹤）

![Git 檔案狀態](https://cdn.zsl0621.cc/2025/docs/areas_upscayl---2025-04-27T15-47-30.webp "Git 檔案狀態")

這張圖解釋了檔案的狀態，我們輔以實際作業流程解釋，讓流程和圖片兩者可以互相對照：

1. **工作目錄 (working directory, 硬碟)** 中的檔案修改後會進入已修改狀態
2. 使用 `git add` 將檔案放進**預存區 (staging area)**
3. 完成程式碼修改後使用 `git commit` 將預存區的檔案提交到**本地儲存庫 (repository)**
4. 想要還原到過往版本時，使用 `git checkout` 把以前的版本簽出[^checkout]，放進「工作目錄」中。

[^checkout]: 官方將 checkout 翻譯為簽出，這個指令的行為是「從儲存庫取出該版本放回硬碟」，用途非常廣泛。

## 關鍵字

初學時關鍵字中英混雜有點難記憶，每個人講的也不太一樣，這裡提供一些關鍵字關係對照：

| 狀態           | 位置                      | 相關指令        |   說明         |
|-------------- |-------------------------- |----------------|-------------- |
| 未追蹤/已修改   | 工作目錄 working directory | `git add`      | 存放到預存區    |
| 已預存         | 預存區 staging area        | `git commit`   | 提交到儲存庫    |
| 已提交         | 儲存庫 repository          | `git checkout`<br/> | 取出到工作目錄  |

## 結尾

現在你已經知道 Git 如何運作了，在本地可以預存檔案，決定要提交時就使用 `git commit` 把預存的檔案放到儲存庫變成正式的提交。所有成員的儲存庫包含遠端儲存庫都是完全相同的鏡像，每個成員可以推送各自的提交到遠端儲存庫以共享進度，以及拉取遠端儲存庫更新的提交到本地儲存庫中。想要回溯歷史時，`git checkout` 就可以從儲存庫簽出舊版檔案，這樣就是一般的作業流程了。

接下來請跳到[一分鐘入門](/basic/one-minute)，序章的其他文章初學者暫時用不到。
