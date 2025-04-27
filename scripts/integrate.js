import fs from "fs-extra";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const rootDir = path.resolve(__dirname, "..");
const vitepressDir = path.join(rootDir, "ripgit");
const docusaurusDir = path.join(rootDir, "main-site");
const vitepressDist = path.join(vitepressDir, ".vitepress", "dist");
const docusaurusBuild = path.join(docusaurusDir, "static");
const targetDir = path.join(docusaurusBuild, "ripgit");

async function integrate() {
  try {
    if (!fs.existsSync(vitepressDist)) {
      throw new Error("VitePress 未正確建構完成，找不到 .vitepress/dist");
    }

    if (fs.existsSync(targetDir)) {
      console.log("移除舊有的 /ripgit 目錄...");
      fs.removeSync(targetDir);
    }

    console.log("搬移 VitePress 至 Docusaurus static...");
    fs.moveSync(vitepressDist, targetDir);

    console.log("\n✅ 整合完成！");
  } catch (err) {
    console.error("❌ 發生錯誤：", err.message);
    process.exit(1);
  }
}

integrate();
