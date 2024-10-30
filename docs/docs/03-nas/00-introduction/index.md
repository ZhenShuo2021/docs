---
title: 從 NAS 學習 Linux
description: 從 NAS 學習 Linux
tags:
  - NAS
  - Linux
keywords:
  - Linux
last_update:
  date: 2024-09-23 GMT+8
  author: zsl0621
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 從 NAS 學習 Linux

首先強調，這是一個新手向的系列文章。

以第一篇文章為例（這篇前言也是和第一篇文章同時寫出來的），其中提到「沙盒」，我國中時網路還不發達，搜尋沙盒會出現沙盒多好多棒有什麼功能，但從來沒人說過沙盒是什麼。第一篇文章也是我剛開始學 Linux 時遇到的問題，在 Google 搜尋 apt apt-get dpkg 遇到同樣問題，於是

1. 會用我覺得方便理解的方式編排
2. 搞清楚讀者是誰，[不要一篇文章要讓新手老手都要過濾內容](/docs/git/preliminaries/introduction)
3. 我超討厭廢話，教學文章絕無廢話
4. 懶的重造輪子，基礎問題問 GPT 最快，會大量使用 GPT 整理的文字並且整理
5. 不要只放指令，那些用久了自然記得起來，但是放一堆指令的我只想馬上關掉網站

可能有人會說鳥哥呢，對我來說那實在是太細節而且太老，以套件安裝為例，dpkg (1994) 現在日常基本上不會用，apt (1998) 對於他撰文的 2003 年才出現五年還是個新東西，apt (2014) 甚至都還沒出現，或者是 OS 選擇，那個年代可能 CentOS 佔額最大，但在 2024 年的今天 [CentOS 已經掛了](https://blog.darkthread.net/blog/centos-is-dead/)，我真的很佩服能看的下去的人。

最後，我的 Linux 學習過程是從 NAS Server 的角度出發，所以我甚至沒用過 Linux GUI，對於想要日常使用的人肯定也會不一樣，請斟酌觀看囉。

<!-- https://www.youtube.com/watch?v=KyADkmRVe0U -->