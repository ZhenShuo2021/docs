---
title: 正則表達式
tags:
  - Python
  - Regex
keywords:
  - Python
  - Regex
last_update:
  date: 2025-03-27T21:20:00+08:00
  author: zsl0621
first_publish:
  date: 2024-09-11T00:00:00+08:00
---

# 正則表達式

```regex
^(.*?)\s*\((\d+)\)(\..+)?$
```

第一次看到正則表達式的感想只有「？？？」，不過講白了只是一套用於描述字符的表達方式。比起看文章死背，我推薦使用 [RegexLearn](https://regexlearn.com/)，非常好馬上學馬上使用，我自己也是看老半天背不起來，後來用他的教學看兩次不到一小時就學會了。

## 入門函式

不造輪子，直接看使用[正規表達式 re](https://steam.oxxostudio.tw/category/python/library/re.html#a01)。

## 入門匹配方式

匹配非保留字符時直接用中括號 `[]` 表示要匹配的字符，例如

```py
text = "ABC 123"
A = re.search("[a-zA-Z]", text).group()   # 'A'
B = re.search("[a-zA-Z]+", text).group()   # 'ABC'
```

也可以加上 `^` 排除字符：

```py
text = "ABC 123"
C = re.search("[^a-zA-Z]", text).group()   # ' '
D = re.search("[^a-zA-Z]+", text).group()   # ' 123'
```

接著介紹如何匹配任意字符，進入滿臉問號的開始。
  
### 字符類型匹配

- `.`：任意字符（不包括換行符）。
- `\d`：任意數字（0-9）。
- `\D`：任意非數字。
- `\w`：任意字母、數字或下劃線（等同[a-zA-Z0-9_]）。
- `\W`：任意非字母、數字或下劃線的字符。
- `\s`：任意空白字符（空格、制表符、換行符）。
- `\S`：任意非空白字符。

使用範例，依序匹配整個字串這樣比較好理解

```py
text = "ABC 123"
# 只匹配單字符
print(re.search(r"\w", text).group())   # 'A'
# 加上"+"匹配多字符
print(re.search(r"\w+", text).group())   # 'ABC'
# 加上"\s"匹配空格
print(re.search(r"\w+\s", text).group())   # 'ABC '
# 加上"\W"匹配字母、數字或下劃線的字符
print(re.search(r"\w+\W", text).group())   # 'ABC '
# 加上"\w+"匹配空格後面的字符
print(re.search(r"\w+\W\w+", text).group())   # 'ABC 123'
```

這就是基本使用方法了。

### 量詞 Quantifier

- `*`：匹配前面的元素 0 次或多次。
- `+`：匹配前面的元素 1 次或多次。
- `?`：匹配前面的元素 0 次或 1 次。
- `{n}`：精確匹配前面的元素 n 次。
- `{n,}`：精確匹配前面的元素 n 次以上。
- `{n,m}`：匹配前面的元素至少 n 次，至多 m 次。

### 邊界設定

- `^`：匹配字符串的開始。
- `$`：匹配字符串的結尾。
- `^XXX$`：整個字串從頭到尾必須匹配。
- `\b`：匹配單詞邊界（例如單詞前後的空格或標點）。
- `\B`：匹配非單詞邊界。
- `XXX(?=YYY)`：正向先行，YYY 匹配的字符「前面」要包含 XXX
- `XXX(?!YYY)`：負向先行，YYY 匹配的字符「前面」不包含 XXX
- `(?<=YYY)XXX`：正向後行，YYY 匹配的字符「後面」要包含 XXX
- `(?<!YYY)XXX`：正向後行，YYY 匹配的字符「後面」不包含 XXX

?向?行可以在 [RegexLearn](https://regexlearn.com/learn/regex101) 的第 45 題找到。

#### 範例：移除電話號碼中的空白

```python
import re

text = "123 456 7890"
result = re.sub(r"\s+", "", text)
print(result)   # 1234567890
```

---

## 高階匹配方式

### 捕獲

- 小括號 `( )`：捕獲一個匹配，以便後續引用
  - EX: `(\d{3})` 匹配並捕獲三位數字
  
- 管道符 `|`：匹配任意一個可能
  - EX: `cat|dog` 匹配 "cat" 或 "dog"

- 非貪婪匹配 `?`：在量詞後加 `?`，使匹配最少次數。正則表達式預設貪婪，它會找最長的匹配

- 小括號加問號 `( )?`：表示可選

#### 範例：取出 HTML 標籤內內容

提取 `"<h1>Hello World</h1>"` \<h1\> 標籤的內容，即 `"Hello World"`。

```python
html = "<h1>Hello World</h1>"
result = re.sub(r"<.*?>(.*?)</.*?>", r"\1", html)
print(result)   # Hello World
```

- `<.*?>`：使用非貪婪模式匹配 HTML 標籤（）。
- `(.*?)`：捕獲標籤中的內容。
- `r"\1"`：取出匹配結果。

#### 範例：移除 URL 的查詢參數

移除所有的查詢參數，只保留基礎 URL （移除問號後面的所有文字）。

```python
url = "https://example.com/page?param1=value1&param2=value2"
result = re.sub(r"\?.*$", "", url)   # 'https://example.com/page'
```

- `\?.*$`：匹配 `?` 後面的所有內容（`$` 表示匹配到行尾）。
- `re.sub` 將匹配到的內容替換成空字符串，從而移除查詢參數。

#### 範例：格式化日期

把日期字串 `"20240908"`格式化成 `"2024-09-08"` 。

```python
text = "20240908"
result = re.sub(r"(\d{4})(\d{2})(\d{2})", r"\1-\2-\3", text)
print(result)   # 2024-09-08
```

- 三個捕獲群組（三個括弧）
- 每個括弧填入匹配規則，分別捕獲年份月份日期
- `r"\1-\2-\3"`：引用捕獲群組，`\1` 是第一個捕獲的組，`\2` 是第二個，`\3` 是第三個。

#### 範例：電子郵件

驗證 `"example@test.com"` 是否符合電子郵件格式。

```python
def validate_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\D{3}$'
    return re.match(pattern, email) is not None

validate_email("example@test.com")   # True
validate_email("example@test.")   # False
validate_email("example@test.s")   # False
validate_email("example@test.sss")   # True
```

- `^`：表示字串開頭
- `[]`：匹配裡面的元素，包含 `\w`, `.`, `-`，
- `[]+`：匹配元素到最長
- `[\w\.-]+`：匹配次級域名
- `\D{3}`：匹配頂級域名，只能是字符不能有數字
- `$`：表示字串結束。

## 結語

regex 是一個神奇的酷東西，我靠 regex 已經交到三個女朋友了，希望大家和我看齊一起學習 regex，謝謝大家。

## 參考資料

[使用正規表達式 re](https://steam.oxxostudio.tw/category/python/library/re.html)
[正規表示式（Regular Expression）](https://hackmd.io/@aaronlife/regular-expression)
