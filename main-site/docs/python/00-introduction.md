---
title: 前言
id: python-hello-page
slug: /introduction
author: zsl0621
last_update:
  date: 2024-09-10T03:07:33+08:00
  author: zsl0621
first_publish:
  date: 2024-09-10T03:07:33+08:00
---

# Python 教學

自學很容易遇到知識零碎不連貫的問題，這份教學的目的就是解決這個問題，清楚說明下一步應該學習什麼，串連零碎的知識。這裡原本是我的知識整理，所以特色是資料來源給好給滿，也因此對於網路上已經重複的內容不會再重複寫一次，只會簡單介紹邏輯，沒人寫過的內容才會有完整教學文章，完整教學會像是這些：

- 全中文最廣泛的[專案管理工具介紹](best-python-project-manager)，沒有任何中文文章教你怎麼選擇，這是唯一一篇
- 全中文最詳細的 [Numba 教學](numba-tutorial-1)，從該不該選擇 Numba 到基礎使用、自動平行化、競爭危害和向量化全部都寫了
- [Numba 效能實測](/python/numba-performance-test)，從基本的三角函數到矩陣計算、訊號處理演算法都測試一次，網路上很多人[亂測試一通](https://stackoverflow.com/a/36533414/26993682)的參考價值都不高

目標讀者主要是已經能寫出基本程式的人，內容都是網路上找不到、缺乏統整、常見的以訛傳訛錯誤等等，現在有這些內容

1. [專案管理工具介紹](best-python-project-manager)比較現存所有的專案管理工具，有高達 19 個
2. [Python uv 教學](uv-project-manager-2)教你最好的專案管理工具 uv 要怎麼用
3. 統整[程式設計原則](programming-principles)，集中在一起介紹而不是講 A 漏 B
4. 搞懂 [Python 核心機制](how-python-works)，包含萬物皆物件、鴨子類型、魔法方法、閉包、垃圾回收、直譯器...
5. [型別註解教學](type-hint)，其實網路上已經很多但是講的都很糟糕，主要是列出優秀的資源並且說明[泛型](type-hint-generic)
6. 完整介紹[多工種類](multitask)還有[Python 中的協程](asynchrony)，這兩個主題的網路資源更是糟糕
7. [完整教學 Numba](numba-tutorial-1) 並且提供相關資源說明[如何優化程式效能](numba-tutorial-2#see-also)
8. [實測 Numba 效能](numba-performance-test)
9. [實測 HTML 解析器效能](html-parser-performance-test)

就是補齊從能動的程式進步到更方便管理、更好、更穩健、更快的程式這一段過程。那麼哪些是還沒寫或者不會寫呢？

- 單元測試：之後補上，真心覺得網路教學看的很吃力，自己最後還是翻文檔或 stackoverflow，希望能寫成最好的中文教學
- 偵錯系統：pdb 研究之後也許可以寫一篇，網路上的 pdb 教學品質也足夠讓我脫穎而出
- 日誌系統：不想獻醜
- 深度學習、容器化、前後端、資料視覺化、科學計算：不想寫這種指定主題的內容
