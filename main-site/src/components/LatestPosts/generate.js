const glob = require("glob");
const path = require("path");
const fs = require("fs");
const moment = require("moment");

// 輸出的 JSON 位置
const latestPosts = "./src/data/latest-posts.json";

// 設定資料來源，包含部落格、文件、備忘錄
const paths = [
  {
    path: "./.docusaurus/docusaurus-plugin-content-blog/default",
    filesPattern: "site-blog-*.json"
  },
  {
    path: "./.docusaurus/docusaurus-plugin-content-docs/default",
    filesPattern: "site-docs-*.json"
  },
  {
    path: "./.docusaurus/docusaurus-plugin-content-docs/pluginForMemo",
    filesPattern: "site-docs-*.json"
  },
  {
    path: "./.docusaurus/docusaurus-plugin-content-docs/pluginForGit",
    filesPattern: "site-docs-*.json"
  },
  {
    path: "./.docusaurus/docusaurus-plugin-content-docs/pluginForPython",
    filesPattern: "site-docs-*.json"
  },
  {
    path: "./.docusaurus/docusaurus-plugin-content-docs/pluginForLinuxCommand",
    filesPattern: "site-docs-*.json"
  },
];

let allItems = [];

if (!fs.existsSync("./.docusaurus")) {
  process.exit(0);
}

paths.forEach(({ path: folderPath, filesPattern }) => {
  const files = glob.sync(path.join(folderPath, filesPattern));

  files.forEach((file) => {
    const rawdata = fs.readFileSync(file);
    const item = JSON.parse(rawdata);

    if (!item || item.draft === true) return;

    let date = item.date ||
      (item.frontMatter?.last_update?.date) ||
      (item.frontMatter?.first_publish?.date);

    if (!date) return;

    const formattedDate = moment(date).format("YYYY-MM-DD HH:mm:ss");
    const yearMonth = moment(date).format("YYYY 年 MM 月");
    const day = moment(date).format("DD");

    allItems.push({
      title: item.title,
      permalink: item.permalink,
      description: item.description || "",
      tags: item.tags || [],
      date: formattedDate,
      yearMonth: yearMonth,
      day: day
    });
  });
});

const latestItems = allItems.sort((a, b) => b.date.localeCompare(a.date));

fs.writeFileSync(latestPosts, JSON.stringify(latestItems, null, 2));
