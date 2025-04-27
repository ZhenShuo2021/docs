---
title: 善用 Git LFS 功能減少儲存庫容量
sidebar_label: Git LFS 減少儲存庫容量
slug: /reduce-size-using-git-lfs
tags:
  - Git
  - Github
  - 教學
keywords:
  - Git
  - Github
  - 教學
last_update:
  date: 2025-04-06T22:11:30+08:00
  author: zsl0621
first_publish:
  date: 2025-04-06T22:11:30+08:00
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

Git 是快照系統，會Git 是快照Git 是快照系統，會藉由差異比較來壓縮空間，這個機制遇到二進制檔案就沒有優勢了，因為很少會有兩個二進制檔案能用簡單的差異來表達，導致檔案有十個版本就要儲存十份，當儲存庫容量過大開發者就要開始使用各種奇怪的 [sparse-checkout 指令](reduce-size-with-sparse-checkout)，所以我們平常在提交二進制檔案時都要小心，壓縮後才上傳也是基本的，最好是根本就不要上傳二進制檔案。

那可不可以把二進制檔案分開儲存呢？Git LFS (large file system) 功能就是把儲存庫的檔案改為指標，真正的檔案儲存在另一個伺服器上大幅減小儲存庫容量，伺服器則是看你用的是哪個雲端服務商，Github 就是用 Github 自己的 LFS 系統，以下簡單介紹幾個特點

- 檔案存儲在專用伺服器上，減少原始 Git 儲存庫的大小
- Git 儲存庫依然會儲存指向這些檔案的指標，指標保存檔案元數據例如哈希值等
- Github/Gitlab 都支援 LFS
- 免費版[容量和每月流量都是 1GB](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-storage-and-bandwidth-usage)

## 安裝

<Tabs>
  <TabItem value="Mac">

  ```bash
  # Homebrew
  brew install git-lfs
  
  # MacPorts
  port install git-lfs
  ```

  完成後貼上 `git lfs install` 初始化。

  </TabItem>

  <TabItem value="Windows">
  
  到[官網](https://git-lfs.com/)下載，或者用 choco 安裝: `choco install git-lfs.install`，完成後使用 `git lfs install` 檢查。

  </TabItem>

  <TabItem value="Linux">

  請見[官方教學](https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md)。
  
  </TabItem>
</Tabs>

## 使用

```sh
git lfs track "*.psd"
git add file.psd
git commit -m 'add file.psd'
```

用法非常簡單和一般的 Git 完全相同，網路上也有很多用法教學，不再贅述。`git lfs ls-files` 可以列出檔案，`git lfs untrack` 可以取消 LFS 追蹤，clone 直接 clone 即可加上 lfs 沒有顯著差異，其餘比較少用的方式就請直接看文檔，因為我也沒用過。

- [將儲存庫中的檔案移至 Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/moving-a-file-in-your-repository-to-git-large-file-storage)
- [從 Git LFS 中刪除文件](https://docs.github.com/en/repositories/working-with-files/managing-large-files/removing-files-from-git-large-file-storage)

## 你適合 Git LFS 嗎？{#cloudflare-r2}

檔案也需要被版本控制系統才適合 Git LFS，如果 LFS 儲存的檔案是網站的圖片那麼 Cloudflare R2 更適合，因為架站的圖片通常沒有必要使用版本控制，大部分情況都是內容過時需要直接更新，不需要保留過時版本。除此之外 Cloudflare 快又便宜：

- 10 GB 容量，每月操作次數寫一百萬，讀一千萬次<u>**免費**</u>
- 20 GB 容量，每月操作次數寫一百萬，讀一千萬次<u>**0.15 美金**</u>
- 50 GB 容量，每月操作次數寫一百萬，讀一千萬次<u>**0.60 美金**</u>
- 1000 GB 容量，每月操作次數寫一百萬，讀一千萬次<u>**15 美金**</u>

> 請到 [R2 Pricing Calculator](https://r2-calculator.cloudflare.com/) 測試

需求是要有一個域名（很便宜一年不用三百塊），有架站的人自己也都有了，唯一缺點在於完全分開控制，需要手動設定上傳圖片，再設定圖片 URL，完整流程會像是 [CloudFlare R2 图床搭建教程](https://www.youtube.com/watch?v=uCHjQp-zH84) 裡面說的一樣。

## 同場加映：圖片壓縮

Mac 預設擷圖是 PNG 很浪費空間，我會使用 imagemagick 把他轉成 webp 節省容量，容量至少減少 3\~4 倍，轉成 AVIF 還可以更小不過有相容性問題，所以大部分還是轉成 webp，指令如下

```sh
magick input.png -sampling-factor 4:2:0 -strip -quality 80 output.webp
```

然而每次都要輸入指令很麻煩，如果是 Mac 用戶可以使用超級方便的 Automator，請見我的 [Automator 文章](/memo/useful-tools/magick-automator)。
