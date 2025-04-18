---
title: Linux 檔案系統基本知識
sidebar_label: 檔案系統基本知識
tags:
  - Linux
  - File System
keywords:
  - Linux
  - File System
last_update:
  date: 2025-03-22T03:28:10+08:00
  author: zsl0621
first_publish:
  date: 2025-03-22T03:28:10+08:00
---

## 檔案目錄結構

FHS 網路上一大堆，沒必要重複寫。

## 檔案類型

Linux 的特色是萬物皆文件，即使是硬碟也會在檔案系統中變成 `/dev/hda` `/dev/sda` 等等。這樣的好處是可以用相同的指令控制硬體設備，簡化系統管理與操作。Linux 檔案系統中的檔案分為以下幾種類型：

- 普通檔案 (-): 包含數據，如文本、圖像或程式碼。
- 目錄 (d): 包含其他檔案和目錄的清單。
- 符號連結 (l): 指向其他檔案或目錄的捷徑。
- 設備檔案
  - 塊設備 (b): 如硬碟，可以隨機訪問數據。
  - 字符設備 (c): 如終端，數據按順序處理。
- 命名管道 (p): 進程間通訊的特殊檔案。
- 套接字 (s): 網路通訊的特殊檔案。

## inode

inode 記錄檔案元資料，和檔案實際內容分開儲存，每個文件都有對應的 inode，我們可以用 `df -i` 查看 inode 總數已用數量，元資料包含

- 檔案大小（字節數）
- 文件位置在硬碟中的位置
- user ID (UID), group ID (GID), 權限
- modification time (mtime), create time (ctime), access time (atime)
- 連結數量，有多少文件名稱指向此 inode

這個特性讓我們在移動檔案時幾乎沒有延遲，因為只需要修改 inode 的元數據就等於移動檔案了，也因此在檔案刪不掉的時可以改為直接刪 inode。

## 軟連結和硬連結

軟連結可以理解為 Windows 的捷徑，訪問軟連結時會自動將操作指向軟連結的實際目標，硬連結則是多個檔案名稱指向同一個 inode，有很多入口，只要還有一個入口存在，檔案就不會被刪除。

硬連結比較少用，不過有一個很好的例子：在 [uv](https://docs.astral.sh/uv) 中所有套件都會被放在快取資料夾，在每個專案下載套件只是從快取資料夾 hard link 過去，這樣不只速度快不需重複下載，多個專案共用相同套件也會不浪費容量。除此之外，硬連結不能跨硬碟，不被 Git 辨識。

軟連結很好理解，但是提供一個小陷阱，不是所有程式都支援軟連結，例如 Nix 把他的設定檔放在 `~/.config` 裡面，而 Nix 本身又使用軟連結，這時如果你的 `~/.config` 是他就會追錯資料夾。

## Logical Volume Manager

一般來說一個硬碟掛載就掛載了，設定完就不能再新增容量，但是 LVM 可以解決這問題。LVM 是把儲存空間虛擬化的技術，由實體硬碟 Physical Volume (PV)，扮演中間人的 Volume Group (VG)，和最後被掛載到系統上的 Logical Volume (LV) 三個組成，概念是透過 VG 作為中間人，只要更改 VG 映射就可以輕鬆修改掛載區空間。

:::info
這是簡化的架構，例如 PV 可以是磁碟分區，完整請見 [How to create a physical volume in Linux using LVM](https://www.redhat.com/en/blog/create-physical-volume)。
:::

以 macOS 為例打開 Disk Utility 看到的卷宗群組就是 VG，裡面有 `Macintosh HD` 和 `Macintosh HD - Data` 兩個卷宗，這個卷宗就是 PV。本文原本只是想搞懂 macOS 這兩個到底是什麼東西，不過既然功課都做了就寫成文章。macOS 的實現一定和 Linux 不一樣，這個段落是說他們概念相同。

## 資料保護觀念

附帶一提，現實上完全沒必要用 LVM，這個結論來自於數據保護的觀點，因為這東西沒有冗餘機制，硬碟又是一新一舊，哪天舊的壞了會讓你連新的都要處理，比手動搬移硬碟麻煩一百倍。檔案系統 btrfs 或 ZFS 都有 pool 和快照功能，資料救援廠商對 btrfs 也很熟悉，不要拿一個很少人用的方式挑戰廠商的技術，就算挑戰成功傷害的也是自己的錢包，更不要說資料救援本來就不保證資料救的回來。

那如果已經頭洗下去用 ext4 怎麼辦？加硬碟後直接掛載使用，把新硬碟當作資料碟，設定跑掉沒差，檔案沒了就是沒了。
