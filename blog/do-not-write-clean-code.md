---
title: 別寫乾淨的程式
authors: zsl0621
keywords:
  - 閱讀心得
tags:
  - 閱讀心得
date: 2024-11-30T00:00:00+08:00
---

最近看到[黑暗執行緒](https://blog.darkthread.net/)關於重構的文章很感興趣，存了一陣子終於有機會搬出來稍微整理一下，本文是個人紀錄和閱讀心得。

程式新手上網查怎麼寫好程式就會看到[如何撰寫乾淨的程式](/blog/how-to-write-clean-code)寫的那些原則，然而很多文章把「必要原則」和「建議原則」寫在一起導致很多誤會，以 DRY 為例：

> <h1>Repeat Yourself</h1>
>
> 1. 工程師 A 觀察到程式碼中存在重複。
> 2. 工程師 A 將這些重複提取出來並賦予它一個名稱，形成新的抽象化，這可能是一個方法，也可能是一個類別。
> 3. 工程師 A 將重複的程式碼替換為新的抽象化，感覺程式碼變得完美無缺後心滿意足地離開。
>
> 時間過去……
>
> 4. 新的需求出現，現有的抽象化幾乎能滿足，但仍需進行少許改動。
> 5. 工程師 B 被指派來實現這項需求，他們希望能保留現有的抽象化，於是通過增加參數和條件邏輯來適應新的需求。
>
> 這樣一來，曾經的通用抽象化開始因應不同情況而表現出不同行為。
>
> 隨著更多需求的出現，這個過程持續重複：
>
> 6. 又來一個工程師 X。
> 7. 又增加一個參數。
> 8. 又新增一個條件判斷。
>
> 最終，程式碼變得難以理解且錯誤頻出。而此時，你正好加入了這個項目，並開始陷入混亂。

<!-- truncate -->

除了黑暗執行緒的文章以外還有以下幾篇文章：

- [閒聊 - 「好程式」跟你想的不一樣! 初讀「重構」有感](https://blog.darkthread.net/blog/refactoring-and-performance/)
- [重構筆記 - 壞味道 (Bad Smell)](https://blog.darkthread.net/blog/refactoring-notes-2-bad-smell/)
- [重構筆記 - .NET 壞味道補充包](https://blog.darkthread.net/blog/refactoring-notes-3/)
- [能抓耗子的就是好貓？閒談程式碼 Anti-Pattern](https://blog.darkthread.net/blog/anti-pattern/)

以及幾篇相關文章：

- [重構-改善既有的程式的設計-第二版 練習與筆記](https://bryanyu.github.io/2018/01/07/RefactorPactice)
- [Goodbye, Clean Code](https://overreacted.io/goodbye-clean-code/)
- [The Wrong Abstraction](https://sandimetz.com/blog/2016/1/20/the-wrong-abstraction)
- [Write code that’s easy to delete, and easy to debug too.](https://programmingisterrible.com/post/173883533613/code-to-debug)
- [Repeat yourself, do more than one thing, and rewrite everything](https://programmingisterrible.com/post/176657481103/repeat-yourself-do-more-than-one-thing-and)

## 重構

### 效能

先說我只是掃過「重構」的電子檔第一章節，自己也沒寫過 JS，只是一個小觀點。

看到黑暗執行緒說成這樣我也很感興趣就去網路上找 PDF 讀了第一章。單純看黑暗執行緒的描述，如果是我寫八成也會想辦法合併迴圈，這裡就要提醒自己「相同等級的時間複雜度沒必要特別優化」，以及「編譯器比自己還聰明」。拿古老的 duff's device 為例，這種神奇的方式[現代編譯器](https://www.youtube.com/watch?v=-WFtkrhzTtg)開 -O3就沒了沒必要搞這些，最後效能提升可能都 negligible。

效能優化問題就像我自己寫的[效能測試](https://docs.zsl0621.cc/docs/python/false-sharing-in-python)一樣，在優化效能之前先搞清楚瓶頸和優化平台、語言等，而不是被假議題騙了。以 [Python 科學運算](/python/numba-tutorial-accelerate-python-computing#see-also)為例，想都不用想就是改用 Numba 或 pybind11，其他都是徒勞，除此之外還要對現代硬體和編譯器有正確認知，例如 [unconditional writes](https://pythonspeed.com/articles/speeding-up-numba/) 這種略為 tricky 的方式就是很好的實現。

總結就是搞清楚任務瓶頸、程式語言、硬體平台和編譯器。

### 我的看法

我也叫語言模型摘要了「重構」此書的幾項重點：

1. **提升可讀性與可理解性**  
不良的程式碼結構會使新成員無法迅速理解系統邏輯，甚至讓經驗豐富的開發者無法有效修改。

2. **減少技術債**  
隨著專案演進，程式碼往往會積累不必要的複雜性或重複邏輯，這些技術債將大幅增加維護成本。

3. **改善軟體穩定性與性能**  
壞程式碼結構可能導致更多的 Bug，甚至在修改時引入新的問題，進而損害產品的穩定性。

4. **促進擴展性**  
良好的程式碼結構能夠輕鬆應對新功能的引入，而壞的結構則可能造成系統崩潰或功能衝突。

除此之外，我不認同 [bryanyu](https://bryanyu.github.io/2018/01/07/RefactorPactice/) 這篇文章所說的「改進軟體設計：一個主要的方向就是消除重複的程式碼」，過度的抽象會導致改一個東西會需要動到其他現有的程式碼，文章後半段會更詳細說明這個問題。在經歷過三個專案後，也認知到重構**應該先預估**預期結果和未來的擴展，不過我目前能力還做不到的預估，現在我則是會考慮

1. 問題的核心是什麼？
2. 可讀性、可維護性、可擴展性
3. 效能

也就是在搞清楚自己的問題後，針對「可讀、可維護、可擴展性」進行修改，並且在修改時提醒自己最開始分析的核心避免改到昏頭轉向，不過對於未來的可維護性和可擴展性方面，目前自身能力不足還沒辦法看到未來情況，也想過可能是因為我自己寫爽的想加啥都是臨時想到根本沒有計畫，沒計畫哪知道未來長怎樣，還有新增的所有功能對我來說都是新工具所以不好預估。

## 可讀性

提升可讀性聽起來簡單但實際上也是有的搞，從基礎的命名規範和一致性，到 SRP 職責劃分、Keep It Simple, Stupid (KISS)、上下文相關性、命名藝術（真的是藝術）、**要不要抽象重複程式碼**、專案生命週期...都有得考量。

### 基礎

- 命名、日誌訊息和錯誤訊息一致性
- 避免魔術數字
- 避免過度封裝
- 清晰易懂的變數命名
- 善用 early return
- 自定的錯誤處理方便定位問題
- 務必使用 linter 和 formatter 協助排版
- 單一職責 SRP: 每個模組、函式或類別只負責一個任務
- 有意義的註釋: 不要寫廢話、盡量寫為什麼而不寫是什麼
- 避免過度嵌套: 根據 Linux 風格指南，超過三層的嵌套代表 doomed，請見[如何優雅地避免程式碼巢狀 | 程式碼嵌套 | 狀態模式 | 表驅動法 |](https://www.youtube.com/watch?v=dzO0yX4MRLM)
- 內聚性: 模組內的成員都在為達成一個清晰、單一的目的或功能而合作。例如負責處理用戶資料的模組，裡面的方法只與用戶資料處理有關，沒有做任何不相關的工作（例如生成報告或處理訂單）
- 對修改封閉: 設計要確保當功能需求變動時可以避免修改現有的程式，而是通過擴展現有系統完成，簡單的例子是使用條件判斷更新就更動到原有程式碼

### 避免奇妙語法

「能抓耗子的就是好貓？閒談程式碼 Anti-Pattern」這篇文章提到奇妙語法的問題，由於沒寫過 JS 所以請出 GPT:

> **簡而言之，作者反對以下兩種寫法：**
>
> 1. **不當使用 `jQuery.map()` 取代 `jQuery.each()` 迭代：** `map()` 的目的是產生 *新陣列*，而非單純迭代。即使 `map()` 可用於迭代，但這 *違反其原意*，造成誤解。僅需迭代時應使用 `each()` 或原生迴圈。
> 2. **不當使用 `Select(o => ...).Count()` 驗證或修改資料：** `Select()` 的目的是產生 *新序列*，`Count()` 是 *計數*。使用它們驗證或在 `Select()` 內 *修改原始資料* 皆 *不當*，嚴重違反其原意，導致後續維護困難。驗證或修改資料應使用 `ForEach()` 或 `foreach()`。

奇妙用法除非有明顯的效能優勢或者明確註解，不然省了行數看起來很爽，結果別人讀要多花一分鐘，更不要說自己以後回來看可能也不記得也需要多讀一分鐘，出 bug 還不知道到底要不要改這裡，簡而言之就是不炫技，不搞怪。

### 码农高天

關於可讀性的另一點是「顯式優於隱式」「語法約束優於邏輯約束」，這兩句話從码农高天偷來的，簡單的範例大概是顯式的寫出 else 比起每次看到還要判斷懷疑一下會不會進入下一行更好。這兩部影片比較推薦觀看，新手中手都適合：

- [【Code Review】十行循环变两行？argparse注意事项？不易察觉的异常处理？](https://www.youtube.com/watch?v=7EQsUOT3NKY)
- [【Code Review】传参的时候有这么多细节要考虑？冗余循环变量你也写过么？](https://www.youtube.com/watch?v=er9MKp7foEQ)

## 不乾淨的程式碼

可維護性是重構的最高原則。

`Write code that’s easy to delete, and easy to debug too.` 這篇文章的標題很清楚就說明「好的程式碼不一定是乾淨的程式碼，而是容易除錯、容易理解其行為和缺陷的程式碼」，程式碼看起來很乾淨不代表他沒有問題，問題反而可能是被隱藏到別的地方，同時也不代表可讀性高，行為應直觀，讓任何開發者都能想出多種變更方式。寫程式的同時要釐清的模糊問題，現在不釐清就是以後除自己的錯，**撰寫易於除錯的程式碼，從意識到未來會忘記這些程式碼開始**。

- [Write code that’s easy to delete, and easy to debug too.](https://programmingisterrible.com/post/173883533613/code-to-debug)
- [Goodbye, Clean Code](https://overreacted.io/goodbye-clean-code/)

## Do Repeat yourself

抽象在幹嘛？以 Python 為例，最直觀的抽象是 `abstractmethod`，就是定義一個模板，子類按規範實作，讓外部只看外觀不用管內部實作。甚至只把重複邏輯包成函式也屬於抽象，總之就是外部只需知道輸入與回傳值。

抽象也可以達到程式設計準則中的 don't repeat yourself (DRY)。那為何說 Do repeat 呢？因為錯誤的抽象比重複還難改，如果寫了一個糟糕、不實際的抽象，或者是沒考慮到未來、對未來支援差、太久以前寫的抽象，要改就不是確認有沒有完整的複製貼上而已。關於這裡因為我還很菜所以各位就請看大老們的文章吧，這裡節錄 `The Wrong Abstraction` 裡面提到的情境：

> 1. 工程師 A 觀察到程式碼中存在重複。
> 2. 工程師 A 將這些重複提取出來並賦予它一個名稱，形成新的抽象化，這可能是一個方法，也可能是一個類別。
> 3. 工程師 A 將重複的程式碼替換為新的抽象化，感覺程式碼變得完美無缺後心滿意足地離開。
>
> 時間過去……
>
> 4. 新的需求出現，現有的抽象化幾乎能滿足，但仍需進行少許改動。
> 5. 工程師 B 被指派來實現這項需求，他們希望能保留現有的抽象化，於是通過增加參數和條件邏輯來適應新的需求。
>
> 這樣一來，曾經的通用抽象化開始因應不同情況而表現出不同行為。
>
> 隨著更多需求的出現，這個過程持續重複：
>
> 6. 又來一個工程師 X。
> 7. 又增加一個參數。
> 8. 又新增一個條件判斷。
>
> 最終，程式碼變得難以理解且錯誤頻出。而此時，你正好加入了這個項目，並開始陷入混亂。

- [Repeat yourself, do more than one thing, and rewrite everything](https://programmingisterrible.com/post/176657481103/repeat-yourself-do-more-than-one-thing-and)
- [Goodbye, Clean Code](https://overreacted.io/goodbye-clean-code/)
- [The Wrong Abstraction](https://sandimetz.com/blog/2016/1/20/the-wrong-abstraction)
- [程式碼中的抽象](https://op8867555.github.io/posts/2021-11-19-abstraction.html)
- [淺談「錯誤的抽象」](https://rickbsr.medium.com/%E6%B7%BA%E8%AB%87-%E9%8C%AF%E8%AA%A4%E7%9A%84%E6%8A%BD%E8%B1%A1-28c0adbf792e)

在 `Goodbye, Clean Code` 裡面提到的「即使程式碼看起來很亂，但是要在裡面加東西比抽象方法簡單多了，正好呼應了 `Write code that’s easy to delete, and easy to debug too.` 裡面的「有時，程式碼本身非常混亂，任何企圖“清理”它的行為反而會帶來更大的問題。在未理解其行為前試圖撰寫乾淨程式碼，結果可能適得其反，無異於召喚出一個難以維護的系統。」

## Law of Demeter 不是「法律」

Law of Demeter (LoD) 指的是不經過多層次的調用，例如 `person.address.country.code` 經過三層的調用取得該人的國籍碼就被視為違反 LoD 原則，目的是避免程式耦合問題。

我的程式並沒有遇到太多這種鏈式調用問題，但是發現反對程式設計原則的文章後覺得很有趣，也去查了有沒有關於反對 LoD 的文章，果然被我找到 [The Law of Demeter Creates More Problems Than It Solves](https://naildrivin5.com/blog/2020/01/22/law-of-demeter-creates-more-problems-than-it-solves.html)，於是放上來作為分享。文章指出以下幾點關鍵問題：

1. Demeter 法則過於簡單化，且被誤解為「避免多於一個點」  
很多人把 Demeter 法則簡化為「程式碼行不能有多於一個點（如 person.address.country.code）」。錯誤的簡化概念反而讓人忽略了法則的真正目的：**降低耦合性**，結果導致過於複雜的封裝與過度抽象，反而降低程式的可讀性與效率。

2. 「法律」命名過於嚴肅  
作者表示把它稱為「法律」會造成問題，因為它並非基於實證，而僅僅是建議，這在英文好的人比較有影響。裡面表示多數程式設計師對其理解過於片面，沒有深入研究原始文獻。盲目遵循 Demeter 法則會導致低效代碼，例如強行使用 Demeter 法則會增加不必要的抽象層，例如添加大量的「代理方法（proxy methods）」來封裝訪問，最終導致程式碼膨脹和難以維護。

3. 解耦與高內聚需要上下文判斷  
應該根據應用領域的上下文來決定解耦程度。文章表示領域核心概念 (core domain concepts) 的穩定性往往比遵循 Demeter 法則更重要。例如對應用程式中的核心結構（如 Person、Address 和 Country）進行合理的耦合通常是可接受的。

4. Demeter 法則忽略了實用性的權衡  
強行消除耦合往往會增加開發成本，例如：需要更多測試、增加程式碼的複雜性，這些額外的抽象帶來的效益可能大於成本。

## 流失率 Code Churn Rate

流失率表示**事後回來修改現有程式的比率**，沒看到正式的定義，只有看到簡單的定義是

> Code Churn Rate = (新增或修改的程式碼行數 + 刪除的程式碼行數) / 總程式碼行數

這是用程式檢查工具才看到的名詞，不知為何幾乎沒什麼人談論到他。根據 [Code Churn Rate: Challenges, Solutions, and Tools for Calculation](https://medium.com/binmile/code-churn-rate-challenges-solutions-and-tools-for-calculation-62f3e8b31fd7) 的說明，一般來說流失率 25% 以下算正常，15% 就屬於高效的運作了。

文章中有列出幾個會出現流失的情況，包含原型設計階段、完美主義、遇到難題、模糊的要求、優柔寡斷的利害關係人合作，五個我中了四個，那流失率高果然也是跑不掉，不過在最後一個專案流失率問題就好很多了。

## 程式碼檢測工具

在 [重構筆記 - 壞味道 (Bad Smell)](https://blog.darkthread.net/blog/refactoring-notes-2-bad-smell/) 提到的問題，使用現代檢查工具可以輕易的避免，目前我主要使用的有幾個：

1. ruff linter: 程式碼品質檢查、確保一致性、可讀性、自動修復、支援 pep8/flake8/Pylint/Pyflakes 等多種規則設定，還會告訴你新語法跟 why better
2. ruff-format: 格式化程式碼，支援 black 格式、內嵌 isort
3. mypy: 靜態型別檢查，確認參數是否符合 type hint，可以減少很多 typo 問題，也可以檢查到某些位置的 code never reach
4. bandit: 安全漏洞檢查
5. pyupgrade: 檢查有沒有用新版 Python 語法
6. pytest/pytest-cov: 單元測試和覆蓋率
7. pre-commit: 預提交自動執行上述指令
8. viztracer: 我老大码农高天開發的 profiler，好用
9. codeclimate: 吃飽太閒的時候會上去看自己的 Issues/Churn/Maintainability 等，流失率就是在這裡學到的，裡面也同樣用 smells 表示有問題的程式碼

## 其他

不在本文標題中，心得也沒有多到可以寫成文章的地步，流水帳描述目前的狀況

1. 版本管理：心情好就打版本標籤，沒規律
2. 錯誤處理：外層攔截特定例外，底層攔截特定例外，其餘沒想法
3. 一致性：包含日誌和錯誤訊息，要怎麼處理一致性還沒想法，想過用裝飾器但感覺不是最佳解
4. 測試策略：沒策略，感覺重要的就單元測試，主功能有整合測試，但是有感受到 CLI 專案整合測試比起單元測試更重要，尤其是我菜鳥階段視野不夠廣的情況下，整合測試能保證至少錯完還是可以動，而單元測試無法保證
5. CI/CD：白嫖 Github 免費流量
6. scope creep（範圍蔓延）：與現在的我無關但我就想放在這
7. 敏捷開發：與現在的我無關但我就想放在這

## 結尾

本篇就是流水帳紀錄過程和看到的文章，大致上可以用「房間稍微有點亂至少行動方便，乾乾淨淨但是反而會造成作什麼都麻煩」概括，或者再加上「避免過早抽象」這個簡易結論。

## Reference

- [閒聊 - 「好程式」跟你想的不一樣! 初讀「重構」有感](https://blog.darkthread.net/blog/refactoring-and-performance/)
- [重構筆記 - 壞味道 (Bad Smell)](https://blog.darkthread.net/blog/refactoring-notes-2-bad-smell/)
- [《先整理一下？個人層面的軟體設計考量》讀後心得分享](https://blog.miniasp.com/post/2025/01/18/Tidy-First-A-Personal-Exercise-in-Empirical-Software-Design-Notes)
- [重構筆記 - .NET 壞味道補充包](https://blog.darkthread.net/blog/refactoring-notes-3/)
- [能抓耗子的就是好貓？閒談程式碼 Anti-Pattern](https://blog.darkthread.net/blog/anti-pattern/)
- [重構-改善既有的程式的設計-第二版 練習與筆記](https://bryanyu.github.io/2018/01/07/RefactorPactice)
- [Write code that’s easy to delete, and easy to debug too.](https://programmingisterrible.com/post/173883533613/code-to-debug)
- [Goodbye, Clean Code](https://overreacted.io/goodbye-clean-code/)
- [The Wrong Abstraction](https://sandimetz.com/blog/2016/1/20/the-wrong-abstraction)
- [Repeat yourself, do more than one thing, and rewrite everything](https://programmingisterrible.com/post/176657481103/repeat-yourself-do-more-than-one-thing-and)
- [程式碼中的抽象](https://op8867555.github.io/posts/2021-11-19-abstraction.html)
- [淺談「錯誤的抽象」](https://rickbsr.medium.com/%E6%B7%BA%E8%AB%87-%E9%8C%AF%E8%AA%A4%E7%9A%84%E6%8A%BD%E8%B1%A1-28c0adbf792e)
- [淺談「重覆程式碼」](https://rickbsr.medium.com/%E6%B7%BA%E8%AB%87-%E9%87%8D%E8%A6%86%E7%A8%8B%E5%BC%8F%E7%A2%BC-fdc45d4990fc)
- [【Code Review】十行循环变两行？argparse注意事项？不易察觉的异常处理？](https://www.youtube.com/watch?v=7EQsUOT3NKY)
- [【Code Review】传参的时候有这么多细节要考虑？冗余循环变量你也写过么？](https://www.youtube.com/watch?v=er9MKp7foEQ)
- [如何優雅地避免程式碼巢狀 | 程式碼嵌套 | 狀態模式 | 表驅動法 |](https://www.youtube.com/watch?v=dzO0yX4MRLM)
- [The Law of Demeter Creates More Problems Than It Solves](https://naildrivin5.com/blog/2020/01/22/law-of-demeter-creates-more-problems-than-it-solves.html)
