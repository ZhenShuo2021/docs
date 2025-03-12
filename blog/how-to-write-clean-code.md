---
title: 如何撰寫乾淨的程式
authors: zsl0621
keywords:
  - 閱讀心得
tags:
  - 閱讀心得
date: 2024-11-30T00:00:00+08:00
---

# 如何撰寫乾淨的程式碼  

關於撰寫乾淨且可讀性高的程式碼的基本知識。

<!-- truncate -->  

碼農高天：CPython 核心開發者，微軟安全團隊員工。

## 原則  

這些原則不是絕對規則，不應該盲目遵守原則，真正的目標是撰寫可維護、可擴展、可讀性高且高效能的程式碼。

1. 可讀性
    - KISS (Keep It Simple, Stupid)
        - 保持程式碼簡單明瞭，避免不必要的複雜性
        - 使用清晰的命名和簡潔的邏輯
        - 避免為炫耀技巧而寫複雜難懂的程式碼
    - 單一職責原則
        - 每個類別或模組應該只有一個改變的理由
    - 程式碼風格一致性
        - 遵循一致的程式碼風格指南 (PEP 8)
        - 使用程式碼格式化工具確保一致性 ([Black](https://blog.kyomind.tw/flake8-and-black/)) ([Ruff](https://blog.kyomind.tw/migrate-to-ruff/))
        - 補充，Python 身為動態語言也可以加上 [linter](https://blog.kyomind.tw/flake8-yapf-setting/) 進行靜態檢查，可參可這三篇文章
    - 撰寫有意義的註釋
        - 解釋程式碼的目的和邏輯，但避免過度註釋
        - 甚至於程式改了註釋記得刪掉也是，真看過有人抱怨被註釋騙
        - 好的程式碼光是用變數和常見邏輯就可讀懂意義，過多的註釋就是廢話或程式碼太複雜！
    - 顯式優於隱式
      - 程式碼的意圖應該明確，避免隱式表達
      - 隱示代表程式語言本身任何的隱示表達方式，例如
        - `from package import *`
        - [`語法約束好於邏輯約束`](https://youtu.be/7EQsUOT3NKY?si=_0m5QidZkAGhSYM2&t=180)
        - [`字典取鍵值`](https://youtu.be/er9MKp7foEQ?si=JZbsHST4aBu2_pis&t=827)
        - [`顯式的寫出 if-else`](https://youtu.be/vbF5M1L2SnU?si=Zao09Wxk0rHwfFmi&t=236) 就算只是補上 else 都比懶惰不寫好
    - **把握可讀性後，會覺得自己的程式碼瞄過去感覺跟 Github 那些大專案有五六分像，別人也可以輕鬆理解自己的程式邏輯**

2. 可維護性
    - DRY (Don't Repeat Yourself)
        - 避免重複程式碼，提高可重用性和可維護性
        - 將重複邏輯提取到函式或類別中
    - 關注點分離 (Separation of Concerns)
        - 將程式分為不同部分，每部分解決單獨問題
        - 提高模組化，降低耦合度
    - 高內聚低耦合
        - 確保模組內部緊密相關，模組間盡量獨立
    - **把握可維護性，可以減少改一個錯兩個的問題**

3. 可擴展性
    - YAGNI (You Aren't Gonna Need It)
        - 只實現當前需要的功能，避免過度設計
        - 保持程式碼靈活性，降低維護成本
    - 組合優於繼承
        - 優先使用組合而非繼承來實現程式碼重用
    - 迪米特法則 (Law of Demeter / Principle of Least Knowledge)
        - 一個物件應對其他物件有最少的了解
        - 減少耦合，提高模組化
    - **把握可擴展性就不會有每次加東西都覺得到處衝突，或者是增加新功能還要修改舊功能的問題**

4. 錯誤處理
    - Fail-fast 原則
        - 儘早暴露錯誤，以便及時發現和解決問題
        - 使用斷言、異常處理和日誌記錄等技術
    - 錯誤處理 (Error Handling)
        - 預期和處理可能發生的錯誤
        - 提供有意義的錯誤訊息

5. 效能與優化
    - 避免過早優化 (Avoid Premature Optimization)
        - 先寫出可讀性高的程式碼，再進行效能優化
        - 只有在確認效能瓶頸後，才進行程式碼優化
    - 平衡 spaghetti code 和 ravioli code
        - 避免過度複雜 (spaghetti) 或過度分割 (ravioli) 的程式碼結構

其他重要原則
    - 單一職責原則 (很重要就再寫一次)
        - 每個類別或模組應該只有一個改變的理由
    - SOLID 原則
        - 複雜，五個原則可以寫成五篇文章，把握好基本原則再看 SOLID 原則
    - 最小驚訝原則 (Principle of Least Astonishment)
        - 用於前端設計，看過就好
        - 設計應符合使用者的直覺預期，提高可用性
    - 版本控制 (Git)
    - 單元測試
    - 自動化測試 (Github Actions CI)

## 基礎知識（必讀）  

這部分絕對不能跳過。  

[Clean Code 實戰之 PHP 良好實踐](https://kylinyu.win/php_best_practice)  
從初學者到進階的乾淨程式碼指南。  

[【Code Review】把 & 當 and 用可是不行！測試寫成這樣也有點離譜哦！](https://www.youtube.com/watch?v=ERosfjjY40Y&list=PLSo-C2L8kdSNr5yUJYhyDArnM4FU9iG1S)  
適合初學者的程式碼審查，展示基本的程式碼優化技巧。  

[如何優雅地避免程式碼巢狀 | 程式碼嵌套 | 狀態模式 | 表驅動法 |](https://www.youtube.com/watch?v=dzO0yX4MRLM)  
介紹減少巢狀結構的方法，包括表驅動法、提前返回、斷言、多型，以及內建函數（filter、sort、group、map、any、all），並探討更好的空指標處理方式。  

[【Python】原來我可以少寫這麼多 for 迴圈！學會之後程式碼都更 Pythonic 了！](https://www.youtube.com/watch?v=8DJ6M3tvnwY)  
介紹 Python 中實用的內建函數，並與手寫 for 迴圈進行效能比較。  

## 乾淨程式碼  

每篇文章都經過精選，內容具參考價值。  

[我與微軟的程式碼規範之爭——局部變數竟然不讓初始化？](https://www.youtube.com/watch?v=cAvAbyadts4)  
討論 C 語言中是否應初始化局部變數，並分析相關錯誤的權衡。  

[【Code Review】傳參的時候有這麼多細節要考慮？冗餘的迴圈變數你也寫過嗎？](https://youtube.com/watch?v=er9MKp7foEQ)  
精選程式碼審查案例。  

[【Code Review】十行迴圈變兩行？argparse 注意事項？不易察覺的異常處理？](https://www.youtube.com/watch?v=7EQsUOT3NKY)  
精選程式碼審查案例。  

[【Code Review】格式、異常處理與多執行緒風險](https://www.bilibili.com/video/BV1iS421Q7Bb)  
審查一個關於 while 迴圈超時控制的有趣專案。  

## 其他知識  

[Win 系統舊代碼導致 CPU 過熱？Google 工程師背鍋 | Google | 微軟 | Chrome | 負優化 | 記憶體 | 系統 | Windows | 程式設計 | CPU](https://www.youtube.com/watch?v=9RjZxB1M1P0)  
過時且不清晰的 Windows API 導致分頁錯誤開銷過大。  

[【Python】聽說 Python 的多執行緒是假的？它真的毫無價值嗎？](https://www.youtube.com/watch?v=1Bk3IpNsvIU)  
深入解析 Python 多執行緒，包括 asyncio、aiohttp。  

[Kotlin 顏值為何遙遙領先 | 不可變變數 | lambda | 語法糖 | 建構函式 | 教程 | 中綴表達式 | val var](https://www.youtube.com/watch?v=iTy13tsi054)  
良好的設計讓唯讀變數比例從 2.9% 提升至 86.3%，輕鬆提升程式碼品質。
