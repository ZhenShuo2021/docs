---
title: 效能分析 Profiling
sidebar_label: 效能分析 Profiling
slug: /profiling
tags:
  - Python
keywords:
  - Python
last_update:
  date: 2024-12-25T11:49:30+08:00
  author: zsl0621
first_publish:
  date: 2024-12-24T21:53:30+08:00
---

<u><b>沒有進行效能分析就不要優化程式碼，都不知道問題在哪裡，如何解決問題？</b></u>

<br/>
<br/>

以 Zsh 啟動速度為例，很多網路文章都是來搞笑的，只知道抱怨速度慢卻沒有進行 profiling，沒頭沒尾的就指控是 Zsh 插件管理器很慢，有沒有想過一個可能，原因是插件太多呢？又有沒有可能八成的延遲都來自兩成的插件呢？甚至問題不是來自插件系統而是自動補全系統呢？如果不先進行 profiling 就會和這些搞笑文章一樣瞎猜一通胡亂優化。

Python 也是一樣，而且不同於 Zsh 有著更好的生態系，請務必在效能優化前分析效能瓶頸。

## 分析工具{#profiling-tools}

profiling 工具可以幫助我們定位效能瓶頸，有幾個定位方式，分別是函式、函式被呼叫次數和行數三個，分析工具大概有這幾種類型：

1. 測試執行時間：timeit/[hyperfine](https://github.com/sharkdp/hyperfine) 或者自己寫腳本
2. 火焰圖：我老大寫的 VizTracer 可直接在 vscode 使用插件，用法請看[老大的影片](https://www.youtube.com/watch?v=xFtEg_e54as&pp=ygUJVml6VHJhY2Vy)，類似的有 Pyinstrument
3. 詳細報告：[scalene](https://adoni.github.io/2022/08/08/python-profile/)/palanteer > py-spy >>>>>> line-profiler/yappi/cProfile
4. 特殊 profiling：Numba 效能分析有人寫了 [profila](https://github.com/pythonspeed/profila)

這樣盤點下來只有三種，簡單測試時間有 timeit 或者自己寫簡單腳本，火焰圖 Flame Graph 用 VizTracer，詳細的報告用 scalene/palanteer，其中 [scalene](https://github.com/plasma-umass/scalene) 寫了很完整的比較表格，也有[py-spy 和 scalene 的比較文章](https://adoni.github.io/2022/08/08/python-profile/)，scalene 功能更強大好用。

除此之外版本管理 (Git) 也必不可少，不可能記得所有版本和對應的效能，請直接開一個分支專門用於優化效能。

## 效能瓶頸來源{#performance-bottleneck}

這個段落簡單列出有哪些效能瓶頸，8020 法則在這裡也適用，八成的效能問題都來自於兩成的程式碼問題，這些問題通常是以下組成：

1. 沒有必要的乾等，常見於 IO 相關任務  
   例如檔案讀寫、網路請求等同步操作，未使用非同步 (async) 或多執行緒，導致程式在等待時無謂地浪費時間。

2. 錯誤的資料結構和工具  
   先找對工具，例如在需要快速查找的場景中應該使用 set 或 dict `O(1)`，而不是用 list 進行線性搜尋 `O(n)`；大型數據不該用內建 list 而是用 numpy；不該用標準庫的 json 而是效率更高的 orjson/[msgspec](https://jcristharif.com/msgspec/benchmarks.html) 等等。

3. 過度迴圈或重複計算  
   執行不必要的計算、未快取結果或是巢狀迴圈等。以快取為例，一個簡單的 lru_cache 就可以[讓你的 fibonacci 加速百倍](https://www.youtube.com/watch?v=DnKxKFXB4NQ)，類似的還有 Short-circuiting、Lazy Evaluation。

4. 多工處理不足或多工程式錯誤  
   現代 CPU 是多核心，程式卻是單線程浪費硬體資源；錯誤則是競爭鎖、false sharing 等相關多工常見問題，這些問題造成多工反而更慢。

5. 外部資源依賴未優化  
   例如頻繁呼叫資料庫查詢而未批次處理，或未使用快取來減少重複請求的開銷。

6. 演算法  
   直接換一個演算法實現比想老半天有用。

7. 輪子問題  
   別自己造輪子，[以矩陣相乘為例](https://stackoverflow.com/questions/36526708/comparing-python-numpy-numba-and-c-for-matrix-multiplication/36533414#36533414)，這個世界上有無數的數學家和聰明的工程師專門研究如何更快，專注做好自己的事，不要重造輪子。

8. 計算密集的優化錯誤  
   不要在 Python 上優化計算密集的任務，徒勞無功，應該使用 numba/pybind11/pyo3。這是一個超級大的主題，Numba 的部分我有寫[教學](numba-tutorial-1)，進一步優化 Numba 效能請看[筆者整理的相關文章](numba-tutorial-2#see-also)還有 [pythonspeed.com](https://pythonspeed.com/)，深度優化需要考慮到硬體層面的執行方式。

這些是主要的瓶頸來源，找到來源才能對症下藥。優化也不是一次性的，每次都是分析效能、定位問題、優化和驗證重複輪迴。優化要注意可讀性，並且注意我們是人類不是編譯器，不要想著取代編譯器的工作。

## 基準測試{#benchmark}

本文雖然不包含如何進行基準測試但是簡單介紹他，因為有效的基準測試可以讓我們知道優化「真正的」提升程度，也釐清自己到底要優化什麼：是縮短使用者感受到的延遲 (Latency)？還是提高系統單位時間內能處理的請求數量 (Throughput)？或是減少記憶體使用量 (Memory Consumption)？還是降低執行時間 (Execution Time)？或是降低 CPU 使用率 (CPU Utilization)？或者是減少 I/O 操作次數？這裡的很多問題其實都不用真正重構，使用懶加載、預先加載、背景處理就可以大幅度改善問題。

不好的基準測試會造成分數和實際感受不同的問題，舉例來說，每個語言模型發佈都宣稱自己的 coding 分數多少，實際叫他寫程式還是一樣人工智障，這是一個很直觀的範例，因為這些廠商的基準測試是用來讓消費者掏出錢包，而不是反應真實情況的；又或者是 [Zsh 啟動速度](/memo/linux/fastest-zsh-dotfile) 問題，[zsh-bench](https://github.com/romkatv/zsh-bench) 就已經表示測試 `zsh -lic 'exit'` 毫無意義，此數值和用戶實際等待時間沒有關係，網路上還是一堆文章拿著錯誤的測試方式來評斷自己的效能優化。還有一個小故事，讀論文時會發現明明都是同一種場景，不同論文卻用不同的基準測試，而且用了不同基準測試卻又不寫換基準的原因，答案很簡單，因為比不贏只好換一個，這是筆者後來發現的小小黑暗面。

總之找到一個好的基準測試才能有效的評估效能問題。
