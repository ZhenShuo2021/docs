import { articlesMeta, projectsMeta } from "./config";
import latestPosts from '@site/src/data/latest-posts.json';

export function getNavigationData() {
  const docsMap = new Map(latestPosts.map(item => [item.permalink, item]));

  const processFrontmatter = (item) => {
    if (item.link?.startsWith("/")) {
      const frontmatter = docsMap.get(item.link) || {};
      const tags = frontmatter.tags 
        ? frontmatter.tags.map(tag => tag.label) 
        : (item.tags || [""]);
      return {
        ...item,
        title: item.title?.trim() || frontmatter.title || "找不到標題",
        description: item.description?.trim() || frontmatter.description || "",
        image: item.image?.trim() || frontmatter.image,
        tags,
      };
    }
    return item;
  };

  return {
    articlesID: articlesMeta.map(processFrontmatter),
    projectsID: projectsMeta.map(processFrontmatter),
  };
}
