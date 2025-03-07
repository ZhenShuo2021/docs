import React, { useState, useEffect, useRef } from "react";
import BrowserOnly from "@docusaurus/BrowserOnly";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";
import ThemedImage from "@theme/ThemedImage";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import BackToTopButton from "@theme/BackToTopButton";

import styles from "./portfolio.module.css";
import { useDocsData, initFloatingElements, scrollToSection, getDisplayedContent } from "./utils";

// import defaultImage from "../../../docs/docs/git/10-preliminaries/data/branch.webp";
const defaultImage = "https://picsum.photos/900/675?grayscale";
const maxTagsShow = 3;

const Tag = ({ children }) => <span className={styles.tag}>{children}</span>;

const Card = React.memo(({ item, type }) => (
  <div className={styles.card}>
    <div className={styles.cardImageContainer}>
      <Link to={item.link}>
        <ThemedImage
          sources={{
            light: item.image || defaultImage,
            dark: item.image || defaultImage,
          }}
          alt={item.title}
          className={styles.cardImage}
          loading="lazy"
        />
      </Link>
      {/* <div className={styles.cardOverlay}>
        <span className={styles.cardType}>{type}</span>
      </div> */}
    </div>

    <div className={styles.cardContent}>
      <h3 className={styles.cardTitle}>
        <Link to={item.link} className={styles.cardTitleLink}>
          {item.title}
        </Link>
      </h3>
      <p className={styles.description}>
        {item.description || ""}
      </p>
      <div className={styles.cardFooter}>
        {item.tags && item.tags.length > 0 && (
          <div className={styles.tagsContainer}>
            {item.tags.slice(0, maxTagsShow).map((tag, index) => (
              <Tag key={index}>{tag}</Tag>
            ))}
          </div>
        )}
      </div>
    </div>
  </div>
));

export default function Portfolio() {
  const [isScrolled, setIsScrolled] = useState(false);
  const { siteConfig } = useDocusaurusContext();
  const docsMap = useDocsData();
  const heroBackgroundRef = useRef(null);

  const { displayedArticles, displayedProjects } = getDisplayedContent(docsMap);

  useEffect(() => {
    const handleScroll = () => setIsScrolled(window.scrollY > 50);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <Layout title="近期活動" description="文章與專案的近期活動">
      <BrowserOnly>
        {() => {
          useEffect(() => {
            const cleanup = initFloatingElements(heroBackgroundRef, styles.heroBgElement);
            return cleanup;
          }, []);

          return (
            <>
              <div className={styles.hero}>
                <div className={styles.heroContent}>
                  <h1 className={styles.mainTitle}>
                    <span className={styles.titleHighlight}>近期活動</span>
                  </h1>
                  <h3 className={styles.subTitle}>最近的文章和專案</h3>
                  <div className={styles.heroActions}>
                    <button className={styles.heroButton} onClick={() => scrollToSection("articles")}>
                      查看文章
                    </button>
                    <button className={styles.heroButton} onClick={() => scrollToSection("projects")}>
                      瀏覽專案
                    </button>
                  </div>
                </div>
                <div ref={heroBackgroundRef} className={styles.heroBackground}></div>
              </div>

              <div className={styles.portfolioContainer}>
                <section id="articles" className={styles.section}>
                  <h2 className={styles.sectionTitle}>✍️ 文章</h2>
                  <div className={styles.grid}>
                    {displayedArticles.map((article, index) => (
                      <Card key={`article-${index}`} item={article} type="文章" />
                    ))}
                  </div>
                </section>
                <section id="projects" className={styles.section}>
                  <h2 className={styles.sectionTitle}>🚀 專案</h2>
                  <div className={styles.grid}>
                    {displayedProjects.map((project, index) => (
                      <Card key={`project-${index}`} item={project} type="專案" />
                    ))}
                  </div>
                </section>
              </div>
              <BackToTopButton />
            </>
          );
        }}
      </BrowserOnly>
    </Layout>
  );
}
