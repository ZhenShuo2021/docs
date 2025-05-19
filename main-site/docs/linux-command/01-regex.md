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

第一次看到正則表達式的感想只有「？？？」，不過講白了只是一套用於描述字符的表達方式。本文原本是教學，但是 [RegexLearn](https://regexlearn.com/) 太強了所以決定刪掉說明變成 cheat sheet，比起看指令死背我更推薦使用 RegexLearn，我自己也是看半天背不起來，後來用他的教學寫兩次一小時就學會了。

regex 是一個神奇的酷東西，我靠 regex 已經交到三個女朋友了，希望大家和我看齊一起學習 regex，謝謝大家。

## Python re 函式

不造輪子，直接看使用[正規表達式 re](https://steam.oxxostudio.tw/category/python/library/re.html#a01)。

## 所有匹配規則

### 字符類型匹配

- `.`：任意字符（不包括換行符）
- `\d`：任意數字（0-9）
- `\D`：任意非數字
- `\w`：任意字母、數字或下劃線（等同[a-zA-Z0-9_]）
- `\W`：任意非字母、數字或下劃線的字符
- `\s`：任意空白字符（空格、制表符、換行符）
- `\S`：任意非空白字符
- `[]`：中括弧裡面匹配字符集合
- `[^aeiou]`：否定匹配字符集合
- `[a-zA-Z]`：匹配英文字符，範圍是 unicode
- `[\p{P}]`：標點符號
- `[\p{Script=Han}]`：CJK 文字
- `[\u4E00-\u9FFF\u3400-\u4DBF]`：中文，包含基本漢字和擴展A區漢字
- `[\u4E00-\u9FFF\u3400-\u4DBF\u20000-\u2A6DF\u2A700-\u2B73F\u2B740-\u2B81F\u2B820-\u2CEAF\u2CEB0-\u2EBEF\uF900-\uFAFF\u2F800-\u2FA1F]`：中文加上一堆擴展罕字
- `[\u3040-\u309F\u30A0-\u30FF]`：日文，只有平假名和片假名
- `[\p{Script=Hiragana}\p{Script=Katakana}]`：日文，同上
- `[\u1100-\u11FF\uAC00-\uD7AF\u3130-\u318F]`：韓文

:::info
`\d` `\w` `\s` 這些在 Linux 中屬於 extended regex (ERE)
:::

### 邊界設定

- `^`：匹配字符串的開始
- `$`：匹配字符串的結尾
- `^XXX$`：整個字串從頭到尾必須匹配
- `\b`：匹配單詞邊界（例如單詞前後的空格或標點）
- `\B`：匹配非單詞邊界
- `XXX(?=YYY)`：正向先行，XXX 匹配的字符「後面」要包含 YYY
- `XXX(?!YYY)`：負向先行，XXX 匹配的字符「後面」不包含 YYY
- `(?<=YYY)XXX`：正向後行，XXX 匹配的字符「前面」要包含 YYY
- `(?<!YYY)XXX`：負向後行，XXX 匹配的字符「前面」不包含 YYY

什麼向什麼行 (lookahead / lookbehind) 可以在 [RegexLearn](https://regexlearn.com/learn/regex101) 的第 45 題找到，舉例來說 `XXX(?= 匹配)` 正向先行只會匹配後面有 `"空格"匹配` 的 `XXX`，其他以次類推。

:::info

lookahead / lookbehind 屬於 PCRE。

:::

### 量詞

- `*`：匹配前面的元素 0 次或多次
- `+`：匹配前面的元素 1 次或多次
- `?`：匹配前面的元素 0 次或 1 次
- `{n}`：精確匹配前面的元素 n 次
- `{n,}`：匹配前面的元素 n 次以上
- `{n,m}`：匹配前面的元素至少 n 次，至多 m 次

### 其餘雜項

- 小括號 `( )`：捕獲群組，以便後續引用
  - EX: `123-45` 使用規則 `(\d{3})-(\d{2})` 會捕獲 `\1` 是 `123`，`\2` 是 `45`

- 小括號 `(?: )`：分組但不捕獲
  - EX: `123-45` 使用規則 `(?:\d{3})-(\d{2})` 會捕獲 `\1` 是 `45`

- 管道符 `|`：匹配任意一個可能
  - EX: `cat|dog` 匹配 "cat" 或 "dog"

- 非貪婪匹配 `?`：在量詞後加 `?`，使匹配最少次數。正則表達式預設貪婪模式，會找最長的匹配

- 小括號加問號 `( )?`：表示可選

### 行內字串排除

這個問題比較常見所以獨立成一個段落，比如說要找包含 `AAA` 但是不包含 `BBB` 的行，使用此 regex:

```regex
^(?=.*AAA)(?!.*BBB).*
```

## 範例

<details>

<summary>取出 HTML 標籤內內容</summary>

提取 `"<h1>Hello World</h1>"` \<h1\> 標籤的內容，即 `"Hello World"`。

```python
html = "<h1>Hello World</h1>"
result = re.sub(r"<.*?>(.*?)</.*?>", r"\1", html)
print(result)   # Hello World
```

- `<.*?>`：使用非貪婪模式匹配 HTML 標籤（）。
- `(.*?)`：捕獲標籤中的內容。
- `r"\1"`：取出匹配結果。

</details>

<details>

<summary>範例：移除 URL 的查詢參數</summary>

移除所有的查詢參數，只保留基礎 URL （移除問號後面的所有文字）。

```python
url = "https://example.com/page?param1=value1&param2=value2"
result = re.sub(r"\?.*$", "", url)   # 'https://example.com/page'
```

- `\?.*$`：匹配 `?` 後面的所有內容（`$` 表示匹配到行尾）。
- `re.sub` 將匹配到的內容替換成空字符串，從而移除查詢參數。

</details>

<details>

<summary>格式化日期</summary>

把日期字串 `"20240908"`格式化成 `"2024-09-08"` 。

```python
text = "20240908"
result = re.sub(r"(\d{4})(\d{2})(\d{2})", r"\1-\2-\3", text)
print(result)   # 2024-09-08
```

- 三個捕獲群組（三個括弧）
- 每個括弧填入匹配規則，分別捕獲年份月份日期
- `r"\1-\2-\3"`：引用捕獲群組，`\1` 是第一個捕獲的組，`\2` 是第二個，`\3` 是第三個。

</details>

<details>

<summary>電子郵件</summary>

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

</details>

## 參考資料

- [使用正規表達式 re](https://steam.oxxostudio.tw/category/python/library/re.html)
- [正規表示式（Regular Expression）](https://hackmd.io/@aaronlife/regular-expression)
