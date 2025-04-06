---
title: 使用 Git Sparse Checkout 只下載部分專案以加速 Clone 速度
sidebar_label: Sparse Checkout 加速克隆
slug: /reduce-size-with-sparse-checkout
tags:
  - Git
  - 教學
keywords:
  - Git
  - 教學
last_update:
  date: 2025-02-11T07:38:25+08:00
  author: zsl0621
first_publish:
  date: 2024-08-22T21:03:25+08:00
---

Clone 大型儲存庫耗時很長又佔空間，例如[樹梅派原始碼](https://github.com/raspberrypi/linux)高達一百二十萬次提交 clone 一次需要長達 15 分鐘，我們也不需要全部檔案和歷史，這時就可以使用 sparse-checkout 以排除指定檔案避免全部下載。

儲存庫過大有兩種原因，分別是 <u>**儲存庫包含大檔案或單純的文件數量過多**</u>，本教學會分別示範並且搭配其他指令同時使用多管齊下，最後包含實際使用範例，包含如何在十秒內克隆完樹梅派儲存庫。

<br />

## TL;DR

本章節是簡單範例所以設定一層，平常建議 1000 起跳方便回滾歷史同時避免奇怪的問題。

### 情境一：文件數量過多，只取出特定目錄

我們以大型 monorepo [openjdk](https://github.com/openjdk/jdk) 實際測試，假設我們開發時只需要用到 `src/base` 和 `src/desktop` 兩個資料夾，指令如下：

```sh
git clone --filter=blob:none --no-checkout --depth=1 --sparse https://github.com/openjdk/jdk; cd jdk

# Git 2.37 之後因為預設啟用 cone 模式不需要這行
# git sparse-checkout init --cone

git sparse-checkout set src/java.base src/java.desktop
git checkout
```

現在我們處於 sparse checkout 的 cone 模式，此模式只支援加入目錄不支援排除目錄，`git sparse-checkout set` 則是將要加入的目錄寫入 `.git/info/sparse-checkout`，最後在 checkout 時根據此規則取出目錄。

<br />

### 情境二：儲存庫包含大檔案，排除指定檔案

剛剛是 cone 模式，此問題則需要 no-cone 模式解決。以 Blowfish 主題為例，這個儲存庫[提交太多圖片](https://github.com/nunocoracao/blowfish/issues/980)導致容量過大。

```sh
git clone --filter=blob:none --no-checkout --depth=1 --sparse https://github.com/nunocoracao/blowfish.git; cd blowfish

# Git 2.37 之後因為預設啟用 cone 模式所以需要這行
git sparse-checkout init --no-cone

git sparse-checkout set '/*' '!exampleSite/*' '!images/*' '!assets/img/*' '!*.png'
git checkout
```

請注意兩種模式的 pattern 語法不相容。no-cone 的和 .gitignore 完全一樣，以上述指令為例，意思是

```
/*               加入全部檔案
!exampleSite/*   排除該目錄
!images/*        排除該目錄
!assets/img/*    排除該目錄
!*.png           排除所有png檔
```

可以看到第一行 `/*` 加上後面的語法就等同於 gitignore 的邏輯，是黑名單的概念；至於 cone 模式的語法則是簡單把白名單的加入資料夾，在 cone 模式之下只能新增要 checkout 的資料夾無法排除。

<br />
<br />

## 詳細說明

大家應該都看過 [The Will Will Web](https://blog.miniasp.com/) 的介紹，雖然很詳細完整但是讀的時候覺得文章花花綠綠的不太好讀，真正提筆發現全都指令很難避免，於是決定列表式寫出來看起來比較清晰，首先是指令參數介紹

1. `--filter=blob:none`: 不要下載 blobs
2. `--depth=1`: 淺克隆，只複製第一層
3. `--shallow-since=<date>`: 淺克隆，限定日期
4. `--no-checkout`: clone 後不把文件放到工作目錄
5. `--sparse`: 設定稀疏檢出 sparse-checkout
6. 上述五個指令可以獨立使用

接下來是對 sparse checkout 本身的介紹

1. cone 代表圓錐，意思是選定一整個資料夾的目錄白名單模式，no-cone 則是和 .gitignore 一樣的黑名單模式
2. [為什麼取名叫圓錐模式？](https://blog.miniasp.com/post/2022/05/17/Down-size-your-Monorepo-with-Git-Sparse-checkouts#:~:text=%E7%82%BA%E4%BB%80%E9%BA%BC%E8%A6%81%E5%8F%96,%E4%BB%A5%E4%B8%8B%E7%9B%AE%E9%8C%84%E7%B5%90%E6%A7%8B%EF%BC%9A)
3. 使用可以直接編輯 `.git/info/sparse-checkout`，`git sparse-checkout set` 就是修改這個文件。編輯後如果不符合 cone 模式 Git 會提醒你設定衝突並且自動退回 no-cone 模式
4. 使用 `git sparse-checkout disable` 回到一般 checkout 模式

:::tip

很重要就再說一次

cone 代表圓錐，意思是選定一整個資料夾的目錄<u>**白名單模式**</u>，no-cone 則是和 .gitignore 一樣的<u>**黑名單模式**</u>
:::

<br/>
<br/>

## 疑難排解

### 應該用 cone 還是 no-cone

取決於儲存庫類型，no-cone 模式只能用於排除大檔案，他的時間複雜度是 `路徑數量*規則數量`，如果儲存庫非常大使用這個方式取出檔案，算上比較的時間可能反而比直接 clone 還慢，但是對於這種問題是大型 binaries 物件的儲存庫則非常適合。

說是這樣說，但是如果儲存庫的路徑不多，用與不用 cone 模式沒有差別。

### 使用 sparse-checkout 無法切換分支

我的解決方式

```sh
git ls-remote --branches
git fetch --depth=1 origin <remote-branch-name>:<local-branch-name>
git checkout <local-branch-name>
```

指定分支對應的遠端就可以進行 checkout。

這個問題問語言模型是沒用的，因為問題不是 sparse-checkout 而是出在 depth=1 上，知乎的方式寫的又臭又長，手動指定分支不就好了，或者設定深度 1000 不會明顯變慢也免得設定分支的麻煩。

### no-cone 模式一定是黑名單嗎

不一定，只是因為其運作方式比較適合黑名單所以用黑名單解釋，順便和白名單對照方便記憶。黑名單的靈感來源是官方文檔就是把他當黑名單用。

### sparse-index 是什麼？

一個實驗性功能，目的是讓 git status 和 git add 速度加快，使用 `git sparse-checkout init --cone --sparse-index` 開啟，可能不兼容外部工具，請見 git core dev [寫的文章](https://github.blog/open-source/git/make-your-monorepo-feel-small-with-gits-sparse-index/)。

<br/>
<br/>

## 實際測試

### 情境一：文件數量過多

[raspberrypi/linux](https://github.com/raspberrypi/linux) 是有一百二十萬提交的大型儲存庫而且檔案數量非常多，使用此指令測試

```sh
REPO=https://github.com/raspberrypi/linux

time git clone -q $REPO
time git clone --depth=1000 $REPO $(basename $REPO)-d1k"
time git clone --depth=1000 --filter=blob:none $REPO $(basename $REPO)-d1k-blob"
time git clone --depth=1000 --filter=blob:none --no-checkout --sparse $REPO $(basename $REPO)-d1k-blob-sp" 
```

設定 sparse-checkout 時假設目標是要處理[這個 Issue](https://github.com/raspberrypi/linux/issues/6037)，於是設定要處理的目錄為：

```sh
git sparse-checkout set drivers/gpio drivers/irqchip drivers/pinctrl drivers/misc drivers/char drivers/platform arch/arm64 include/linux kernel/irq kernel/printk
```

測試結果如圖所示，sparse 方式雖然沒加上後續的 checkout 時間 9 秒，但是總計 13 秒還是非常快。

![clone time](data/reduce-clone-size-linux.webp)

### 情境二：儲存庫包含大檔案

[xuejianxianzun/PixivBatchDownloader](https://github.com/xuejianxianzun/PixivBatchDownloader) 是一個 Chrome 擴充功能的儲存庫，使用此指令測試

```sh
REPO=https://github.com/xuejianxianzun/PixivBatchDownloader

time git clone -q $REPO
time git clone -q --filter=blob:none $REPO "$(basename $REPO)-blob"
time git clone -q --filter=blob:none --no-checkout $REPO "$(basename $REPO)-blob-nc"
time git clone -q --filter=blob:none --no-checkout --depth=1 $REPO "$(basename $REPO)-blob-nc-d1" 
time git clone -q --filter=blob:none --no-checkout --depth=1 --sparse $REPO "$(basename $REPO)-blob-nc-d1-sp" 
```

設定 sparse-checkout 時假設我們排除關於文檔的目錄，在 `.git/info/sparse-checkout` 裡面新增這些規則

- /dist/
- /notes/
- !/notes/*/
- /src/

測試結果如圖

![clone time](data/reduce-clone-size-pixiv.webp)

欸不是說不能排除目錄嗎？這是撰文途中意外試到的一個小 bug，他只能排除所有子目錄，並且如果使用 `git sparse-checkout set '!/notes/*/'` 會報錯，所以請知悉<u>**這是 bug 不是 feature，只能排除所有子目錄不能排除指定目錄**</u>！！！

> 附帶一提，為什麼被測試的指令跟情境一有微妙的差別呢，因為我忘了，然後我懶的改。

## 參考資料

- https://git-scm.com/docs/git-sparse-checkout
- https://ryanagibson.com/posts/shrink-git-repo
- https://blog.miniasp.com/post/2022/05/17/Down-size-your-Monorepo-with-Git-Sparse-checkouts
