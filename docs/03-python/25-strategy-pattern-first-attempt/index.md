---
title: 初嘗策略模式
description: 第一次撰寫策略模式的紀錄
tags:
  - Programming
  - Python
keywords:
  - Programming
  - Python
last_update:
  date: 2024-10-29T16:29:33+08:00
  author: zsl0621
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 初嘗策略模式

> 紀錄寫爬蟲程式遇到的策略模式問題

原本只是想寫個簡單寫個爬蟲，想說順便學著使用各種程式碼品質工具，例如 ruff linter （語法檢查）和 mypy static typing （靜態類型檢查），然後為了解決靜態類型問題衍生出更多問題，就有了這篇文章。

## 問題描述
一開始很簡單的實現，這個 class，在 `scrape_link` 中基於 `is_album_list` 決定要調用 `_process_album_list_links` 還是 `_process_album_image_links`，每次調用會把該輸入的 URL 翻頁翻到底並擷取每頁結果：如果在相簿列表，獲取所有相簿網址才開始爬取相簿，`scrape_link` 回傳網址列表 `list[str]`；否則回傳圖片網址和檔名列表 `list[tuple[str, str]]`。會這樣寫的原因是該網站只有兩種類型頁面，相簿列表和相簿本身，使用相同翻頁方式，所以覺得沒必要分成兩個 class method 來寫。

但是使用 mypy 檢查時抱怨 incompatible type 的問題就由此引發：

```sh
error: Argument 2 to "_process_album_list_links" of "LinkScraper" has incompatible type "list[str] | list[tuple[str, str]]"; expected "list[str]"  [arg-type]
error: Argument 2 to "_process_album_image_links" of "LinkScraper" has incompatible type "list[str] | list[tuple[str, str]]"; expected "list[tuple[str, str]]"  [arg-type]
```

```py
class LinkScraper:
    """Scrape logic."""

    def __init__(self, web_bot, dry_run: bool, download_service, logger: logging.Logger):
        # ...初始化

    def scrape_link(
        self, url: str, start_page: int, is_album_list: bool
    ) -> list[str] | list[tuple[str, str]]:
        """Scrape pages for links starting from the given URL and page number.

        Args:
            url (str): Initial URL to scrape, which can be an album list or an album page.
            start_page (int): The page number to start scraping from.
            is_album_list (bool): Indicates if the URL is an album list page.

        Returns:
            page_result (list[str] | list[tuple[str, str]]): A list of URLs if is_album_list=True; 
            otherwise, a list of (URL, filename) tuples.
        """
        page_result: list[str] | list[tuple[str, str]] = []
        page = start_page
        consecutive_page = 0
        max_consecutive_page = 3
        alt_ctr = 0
        xpath_page_links = XPATH_ALBUM_LIST if is_album_list else XPATH_ALBUM

        while True:  # each loop turns a page
            full_url = LinkParser.add_page_num(url, page)
            html_content = self.web_bot.get_html(full_url)
            tree = LinkParser.parse_html(html_content, self.logger)
            
            # ...輸入檢查

            if is_album_list:
                self._process_album_list_links(page_links, page_result, page)
            else:
                self._process_album_image_links(page_links, page_result, alt_ctr, tree, page)

            if page >= LinkParser.get_max_page(tree):
                self.logger.info("Reached last page, stopping")
                break

            page += 1
            consecutive_page += 1
        return page_result

    def _process_album_list_links(
        self,
        page_links: list[str],
        page_result: list[str],     # 注意這裡和下面不同！！！
        page: int,
    ):
        """Process and collect album URLs from list page."""
        page_result.extend([BASE_URL + album_link for album_link in page_links])
        self.logger.info("Found %d images on page %d", len(page_links), page)

    def _process_album_image_links(
        self,
        page_links: list[str],
        page_result: list[tuple[str, str]],     # 注意這裡和上面不同！！！
        alt_ctr: int,
        tree: html.HtmlElement,
        page: int,
    ):
        """Handle image links extraction and queueing for download."""
        alts: list[str] = tree.xpath(XPATH_ALTS)

        if len(alts) < len(page_links):
            missing_alts = [str(i + alt_ctr) for i in range(len(page_links) - len(alts))]
            alts.extend(missing_alts)
            alt_ctr += len(missing_alts)

        page_result.extend(zip(page_links, alts))

        # Download file
        if not self.dry_run:
            album_name = self.extract_album_name(alts)
            image_links = list(zip(page_links, alts))
            self.download_service.add_download_task(album_name, image_links)  # add task to queue
        self.logger.info("Found %d images on page %d", len(page_links), page)

    @staticmethod
    def extract_album_name(alts: list[str]) -> str:
        # ...Find the first non-digits element
```

想到的解決方式有這幾個

1. 使用 isinstance 檢查，但是如果中間跨函式 mypy 一樣會抱怨，而且 list 包 tuple 要每項檢查很繁瑣。
2. 使用工廠模式增加一個新的 method 檢查，想了十秒發現根本沒用，現在不就是同樣意思。
3. page_result 改成 list[Any]。
4. 不要寫 type hint 掩耳盜鈴。

同時也想到，如果該網站擴充新的頁面類型，以上方法基本只剩掩耳盜鈴法有用，現在的 scrape_link 結構也會造成維護困難，於是需要更好的解決方式。

## 解決方式
終於進入策略模式，但是現在程式碼改太多了和原本的幾乎認不出是同一個人，這裡就給虛擬碼：


<Tabs>
  <TabItem value="LinkScraper" label="策略模式">
 
```py

class LinkScraper:
    # Defines the mapping from string to scraping method.
    SCRAPE_TYPE: ClassVar[dict[str, str]] = {
        "ALBUM_LIST": "album_list",
        "ALBUM_IMAGE": "album_image",
    }

    def __init__(self):
        # 在這裡初始化所有策略
        self.strategies: dict[str, ScrapingStrategy] = {
            self.SCRAPE_TYPE["ALBUM_LIST"]: AlbumListStrategy(runtime_config, base_config, web_bot),
            self.SCRAPE_TYPE["ALBUM_IMAGE"]: AlbumImageStrategy(
                runtime_config, base_config, web_bot
            ),
        }


    def _scrape_link(
        self,
        url: str,
        start_page: int,
        scraping_type: str,
        **kwargs,
    )
        # 根據字串選擇使用哪種策略
        strategy = self.strategies[scraping_type]
        while True:
            # ...省略

            # 原版程式碼
            # if is_album_list:
            #     self._process_album_list_links(page_links, page_result, page)
            # else:
            #     self._process_album_image_links(page_links, page_result, alt_ctr, tree, page)

            # 新版程式碼
            strategy.process_page_links(page_links, page_result, tree, page)

```



  </TabItem>

  <TabItem value="AlbumListStrategy" label="策略們">
  
```py

class AlbumListStrategy(ScrapingStrategy[AlbumLink]):
    """Strategy for scraping album list pages."""

    def get_xpath(self) -> str:
        return XPATH_ALBUM_LIST

    def process_page_links(
        self,
        page_links: list[str],
        page_result: list[AlbumLink],
        tree: html.HtmlElement,
        page: int,
        **kwargs,
    ) -> None:
        page_result.extend([BASE_URL + album_link for album_link in page_links])
        self.logger.info("Found %d albums on page %d", len(page_links), page)


class AlbumImageStrategy(ScrapingStrategy[ImageLink]):
    """Strategy for scraping album image pages."""

    def __init__(self, runtime_config: RuntimeConfig, base_config: Config, web_bot):
        super().__init__(runtime_config, base_config, web_bot)
        self.dry_run = runtime_config.dry_run
        self.alt_counter = 0

    def get_xpath(self) -> str:
        return XPATH_ALBUM

    def process_page_links(
        self,
        page_links: list[str],
        page_result: list[ImageLink],
        tree: html.HtmlElement,
        page: int,
        **kwargs,
    ) -> None:
        alts: list[str] = tree.xpath(XPATH_ALTS)

        # Handle missing alt texts
        if len(alts) < len(page_links):
            missing_alts = [str(i + alt_ctr) for i in range(len(page_links) - len(alts))]
            alts.extend(missing_alts)
            alt_ctr += len(missing_alts)

        page_result.extend(zip(page_links, alts))

        # Download file
        if not self.dry_run:
            album_name = self.extract_album_name(alts)
            image_links = list(zip(page_links, alts))
            self.download_service.add_download_task(album_name, image_links)  # add task to queue
        self.logger.info("Found %d images on page %d", len(page_links), page)

    @staticmethod
    def _extract_album_name(alts: list[str]) -> str:
        album_name = next((alt for alt in alts if not alt.isdigit()), None)
        if album_name:
            album_name = re.sub(r"\s*\d*$", "", album_name).strip()
        if not album_name:
            album_name = BASE_URL.rstrip("/").split("/")[-1]
        return album_name

```

  </TabItem>

</Tabs>


新版程式碼在每次呼叫 `_scrape_link` 時都根據輸入字串選擇要使用的解析方式，省略了 if-else 條件判斷，也減輕 `LinkScraper` 負擔，把工作侷限程式碼的上下文接口，調用策略，每個策略各自處理該如何解析 html 檔案，可以輕易新增或移除策略。除此之外策略模式的好處還有如果遇到複雜的頁面，需要更多子函式來處理，也不用把分散的 function 全部都匯集到一個 function 中，可以都在各自的 xxxStratedy 類別自行管理。

---

雖然策略模式到這邊就結束了，但是最開始的 mypy 型別檢查的問題好像還沒解決欸？細心的讀者可能已經發現 `_scrape_link` 變成私有函數，這邊使用幾個緩衝 method 解決型別的問題，暫時只能想到這個解決方法，未來如果找到更好方式再更新。

```py
def buffer_album_list(
    self,
    url: str,
    start_page: int,
    **kwargs
    ) -> list[str]:
    """Entry and buffer method for album list scraping."""
    return self._scrape_link(url, start_page, self.SCRAPE_TYPE["ALBUM_LIST"], **kwargs)

def buffer_album_images(
    self, 
    url: str, 
    start_page: int,
    **kwargs
    ) -> list[tuple[str, str]]:
    """Entry and buffer method for Album images scraping."""
    return self._scrape_link(url, start_page, self.SCRAPE_TYPE["ALBUM_IMAGE"], **kwargs)
```


## 心得感想

沒想到修復 mypy 會用到策略模式，除此之外還學了 typing 工具，例如 ClassVar, Generic, TypeAlias, TypeVar。其中還發現了模板方法模式、外觀模式，感覺滿硬要的，正常人寫程式自動就會變成這些模式，沒必要去記這些，死記這些也沒用，就像以前電磁學第一大題都是名詞解釋，考完根本沒有理解核心，不過也是因為電磁學太難需要送分就是了。另外，平衡 spaghetti code 和 ravioli code，以及程式優化的時機也是個學問，太早優化後面要改更痛苦，太晚優化中間開發很卡。

寫那麼長，一言以蔽之就是實現多種不同實作，透過類別封裝，以單一個變數作為入口點，再看你想怎麼調用這個入口，我是以字串調用，下面語言模型生成的範例放在列表裡面調用。

> 策略模式我一開始看完也是有點懵，如果還是看不太懂，這裡有一個叫 AI 生成的簡單範例，對購物車的商品進行不同類型的價格計算：

<details>
<summary> Claude 3.5 Sonnet#2024/10</summary>

<Tabs>
<TabItem value="沒有策略模式" label="沒有策略模式">

```py
class Item:
    def __init__(self, price):
        self.price = price

class ComplexShoppingCart:
    PERCENTAGE_DISCOUNT_THRESHOLD = 50  # 超過50元可以打折
    PERCENTAGE_DISCOUNT_RATE = 0.9      # 打9折
    BULK_DISCOUNT_THRESHOLD = 3         # 3件以上打折
    BULK_DISCOUNT_RATE = 0.85          # 打85折
    MEMBER_DISCOUNT_RATE = 0.95        # 會員95折
    TAX_RATE = 0.08                    # 8%稅率

    def __init__(self):
        self.items = []
        # 控制開關
        self.enable_percentage_discount = False
        self.enable_bulk_discount = False
        self.enable_member_discount = False
        self.enable_tax = False
        
    def add_item(self, item):
        self.items.append(item)
        
    def remove_item(self, item):
        self.items.remove(item)
        
    def clear_cart(self):
        self.items = []
        
    # 設置各種折扣的開關
    def set_percentage_discount(self, enabled=True):
        self.enable_percentage_discount = enabled
        
    def set_bulk_discount(self, enabled=True):
        self.enable_bulk_discount = enabled
        
    def set_member_discount(self, enabled=True):
        self.enable_member_discount = enabled
        
    def set_tax(self, enabled=True):
        self.enable_tax = enabled
        
    def get_subtotal(self):
        """計算未折扣前的總金額"""
        return sum(item.price for item in self.items)
    
    def calculate_percentage_discount(self, price):
        """計算單品折扣"""
        if self.enable_percentage_discount and price > self.PERCENTAGE_DISCOUNT_THRESHOLD:
            return price * self.PERCENTAGE_DISCOUNT_RATE
        return price
    
    def calculate_bulk_discount(self, price):
        """計算批量折扣"""
        if self.enable_bulk_discount and len(self.items) >= self.BULK_DISCOUNT_THRESHOLD:
            return price * self.BULK_DISCOUNT_RATE
        return price
    
    def calculate_member_discount(self, price):
        """計算會員折扣"""
        if self.enable_member_discount:
            return price * self.MEMBER_DISCOUNT_RATE
        return price
    
    def calculate_tax(self, price):
        """計算含稅價格"""
        if self.enable_tax:
            return price / (1 - self.TAX_RATE)
        return price
    
    def get_total(self):
        """計算最終總價"""
        if not self.items:
            return 0
            
        total = 0
        for item in self.items:
            item_price = item.price
            
            # 依序套用各種折扣
            # 注意：順序會影響最終價格
            item_price = self.calculate_percentage_discount(item_price)
            item_price = self.calculate_bulk_discount(item_price)
            item_price = self.calculate_member_discount(item_price)
            item_price = self.calculate_tax(item_price)
            
            total += item_price
            
        return total
    
    def print_receipt(self):
        """印出詳細的收據"""
        if not self.items:
            print("購物車是空的")
            return
            
        print("\n=== 購物明細 ===")
        print(f"商品數量: {len(self.items)}")
        print(f"原始總價: ${self.get_subtotal():.2f}")
        
        # 打印啟用的折扣
        if self.enable_percentage_discount:
            print(f"* 單品折扣: 超過${self.PERCENTAGE_DISCOUNT_THRESHOLD}打{self.PERCENTAGE_DISCOUNT_RATE*100}折")
        if self.enable_bulk_discount:
            print(f"* 批量折扣: {self.BULK_DISCOUNT_THRESHOLD}件以上打{self.BULK_DISCOUNT_RATE*100}折")
        if self.enable_member_discount:
            print(f"* 會員折扣: 打{self.MEMBER_DISCOUNT_RATE*100}折")
        if self.enable_tax:
            print(f"* 稅率: {self.TAX_RATE*100}%")
            
        print(f"最終總價: ${self.get_total():.2f}")
        print("================")

# 使用範例
def run_example():
    # 建立購物車
    cart = ComplexShoppingCart()
    
    # 添加商品
    cart.add_item(Item(60))  # 高於單品折扣門檻
    cart.add_item(Item(40))
    cart.add_item(Item(50))  # 總共3件，可以享受批量折扣
    
    # 情境1: 只啟用單品折扣
    print("\n情境1: 只有單品折扣")
    cart.set_percentage_discount(True)
    cart.print_receipt()
    
    # 情境2: 加上批量折扣
    print("\n情境2: 單品折扣 + 批量折扣")
    cart.set_bulk_discount(True)
    cart.print_receipt()
    
    # 情境3: 再加上會員折扣
    print("\n情境3: 單品折扣 + 批量折扣 + 會員折扣")
    cart.set_member_discount(True)
    cart.print_receipt()
    
    # 情境4: 最後加上稅金
    print("\n情境4: 所有折扣 + 稅金")
    cart.set_tax(True)
    cart.print_receipt()

# 執行範例
run_example()
```

</TabItem>

<TabItem value="使用策略模式" label="使用策略模式">

```py
class PricingStrategy:
    def calculate_price(self, item_price, items):
        return item_price

class PercentageDiscountStrategy(PricingStrategy):
    def calculate_price(self, item_price, items):
        if item_price > 50:
            return item_price * 0.9
        return item_price

class BulkDiscountStrategy(PricingStrategy):
    def calculate_price(self, item_price, items):
        if len(items) >= 3:
            return item_price * 0.85
        return item_price

class MemberDiscountStrategy(PricingStrategy):
    def calculate_price(self, item_price, items):
        return item_price * 0.95

class TaxStrategy(PricingStrategy):
    def calculate_price(self, item_price, items):
        return item_price / 0.92  # 8% 稅率

class ImprovedShoppingCart:
    def __init__(self):
        self.items = []
        self.pricing_strategies = []
        
    def add_item(self, item):
        self.items.append(item)
        
    def add_pricing_strategy(self, strategy):
        self.pricing_strategies.append(strategy)
        
    def remove_pricing_strategy(self, strategy):
        self.pricing_strategies.remove(strategy)
        
    def get_total(self):
        total = 0
        for item in self.items:
            item_price = item.price
            # 依序應用每個策略
            for strategy in self.pricing_strategies:
                item_price = strategy.calculate_price(item_price, self.items)
            total += item_price
        return total

# 使用範例
class Item:
    def __init__(self, price):
        self.price = price

# 使用改進後的購物車
cart = ImprovedShoppingCart()

# 添加商品
cart.add_item(Item(60))
cart.add_item(Item(40))
cart.add_item(Item(50))

# 可以同時使用多個策略
cart.add_pricing_strategy(PercentageDiscountStrategy())  # 超過50元打9折
cart.add_pricing_strategy(BulkDiscountStrategy())       # 買3件以上打85折
cart.add_pricing_strategy(TaxStrategy())                # 加稅

# 計算總價
print(f"Total with multiple strategies: {cart.get_total():.2f}")

# 移除某個策略
cart.remove_pricing_strategy(BulkDiscountStrategy())
print(f"Total after removing bulk discount: {cart.get_total():.2f}")
```

</TabItem>
</Tabs>

</details>


## 後話

寫的當下就覺得有點 over-design 了，現在想想確實如此，一開始直接寫 type-hint 是 list 就好，把 type-hint 寫的這麼詳細反而浪費了 Pyhton 方便的特性，尤其是在這麼小的項目上，不過這本來就是一個練習專案，當作練習剛剛好囉。