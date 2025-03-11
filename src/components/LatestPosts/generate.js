const glob = require("glob");
const path = require("path");
const fs = require("fs");
const moment = require("moment");

// 設定資料來源，包含部落格、文件、備忘錄
const paths = [
  {
    path: "./.docusaurus/docusaurus-plugin-content-blog/default",
    filesPattern: "site-blog-*.json",
    sourceType: "blog"
  },
  {
    path: "./.docusaurus/docusaurus-plugin-content-docs/default",
    filesPattern: "site-docs-*.json",
    sourceType: "docs"
  },
  {
    path: "./.docusaurus/docusaurus-plugin-content-docs/pluginForMemo",
    filesPattern: "site-docs-*.json",
    sourceType: "memo"
  },
  {
    path: "./.docusaurus/docusaurus-plugin-content-docs/pluginForGit",
    filesPattern: "site-docs-*.json",
    sourceType: "git"
  },
  {
    path: "./.docusaurus/docusaurus-plugin-content-docs/pluginForPython",
    filesPattern: "site-docs-*.json",
    sourceType: "git"
  },
];

// 輸出的 JSON 位置
const latestPostsList = "./src/components/LatestPosts/latest-posts.json";

let allItems = [];

paths.forEach(({ path: folderPath, filesPattern, sourceType }) => {
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
      day: day,
      source: sourceType
    });
  });
});

const latestItems = allItems.sort((a, b) => b.date.localeCompare(a.date));

fs.writeFileSync(latestPostsList, JSON.stringify(latestItems, null, 2));
