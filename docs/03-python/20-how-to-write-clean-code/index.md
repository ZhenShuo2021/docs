---
title: 寫出優雅程式
description: How to Write Clean Code
tags:
  - Programming
  - Python
keywords:
  - Programming
  - Python
last_update:
  date: 2024-10-01 GMT+8
  author: zsl0621
---

# How to Write Clean Code
Collections of useful infos on the internet for writing clean, readable code.  
码农高天: Core dev of CPython, works in Microsoft kernel security team.   

## Principle
These guides are not absolute rules, you should remember that the final purpose is writing a maintainable, extendable, readable and high-performance code.

> This list is organized by myself, the Chinese text is summarized by GPT. 
1. **KISS (Keep It Simple, Stupid)**: 簡單就是美。複雜的程式碼難以理解、維護和除錯。盡可能保持程式碼簡潔易懂。
2. **DRY (Don't Repeat Yourself)**: 避免重複的程式碼。重複的程式碼難以維護，因為對一處的更改需要在多個地方進行。
3. **YAGNI (You Aren't Gonna Need It)**: 避免過度設計。只在真正需要時才添加功能，不要提前實現你認為*可能*會需要的功能。
4. **Separation of Concerns (SoC)**: 將程式分為不同的部分，每個部分都解決一個單獨的問題。這有助於提高程式碼的可維護性和可重用性。
5. **Principle of Least Knowledge (最少知識原則)**: 一個物件應該對其他物件儘可能少的了解。這有助於降低耦合度，使程式碼更易於維護和修改。
6. **Fail-fast**: 儘早暴露錯誤，以便及時發現和解決問題。這可以通過使用斷言、異常處理和日誌記錄等技術來實現。
7. **Coding conventions**: 遵循一致的程式碼風格和命名約定。
8. **3 levels of indentation**: Linux Kernel Coding Guide: *The answer to that is that if you need more than 3 levels of indentation, you’re screwed anyway, and should fix your program.*
9. **SOLID principle**
10. **Balance between spaghetti code and ravioli code**

## Basic Knowledge (Essential)
You should never skip this part.

[如何優雅地避免程式碼巢狀 | 程式碼嵌套 | 狀態模式 | 表驅動法 |](https://www.youtube.com/watch?v=dzO0yX4MRLM)  
Introduces methods for reducing code nesting, including table methods, early returns, assertions, polymorphism, and useful built-in functions (filter, sort, group, map, any, all). Also covers better null pointer handling.

[【python】原来我可以少写这么多for loop！学会之后代码都pythonic了起来](https://www.youtube.com/watch?v=8DJ6M3tvnwY)  
Introduces useful built-in functions in Python and compares their performance to hand-crafted for loops.

## Clean code
Every article is carefully curated and informative.

[我与微软的代码规范之争——局部变量竟然不让初始化？](https://www.youtube.com/watch?v=cAvAbyadts4)  
Discusses whether to initialize local variables in C and explains the tradeoffs related to code errors.   

[【Code Review】传参的时候有这么多细节要考虑？冗余循环变量你也写过么？](https://youtube.com/watch?v=er9MKp7foEQ)  
SELECTED code review

[【Code Review】十行循环变两行？argparse注意事项？不易察觉的异常处理？](https://www.youtube.com/watch?v=7EQsUOT3NKY)  
SELECTED code review

[【Code Review】格式，异常处理和多线程的风险](https://www.bilibili.com/video/BV1iS421Q7Bb)  
Review code for an interesting project about controlling while loop time out.

## Other Knowledge

[Win系統舊代碼導致CPU干冒煙？谷歌程序員慘背鍋 | 谷歌 | 微軟 | Chrome | 負優化 | 內存 | 系統 | Windows | 程序員 | CPU](https://www.youtube.com/watch?v=9RjZxB1M1P0)  
Too old and not clear Windows API causes page fault overhead.

[【python】听说Python的多线程是假的？它真的没有存在的价值么？](https://www.youtube.com/watch?v=1Bk3IpNsvIU)  
Explains multi-threading of Python, including asyncio, aiohttp.

[Kotlin颜值为啥遥遥领先 | 不可变变量 | lambda | 语法糖 | 构造函数 | 教程 | 中缀表达式 | val var](https://www.youtube.com/watch?v=iTy13tsi054)  
How a good design improves readability and ratio of read-only variables (2.9% to 86.3% without pain).