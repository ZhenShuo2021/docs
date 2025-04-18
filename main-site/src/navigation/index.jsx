import React, { useRef, useEffect } from "react";
import BrowserOnly from "@docusaurus/BrowserOnly";
import Layout from "@theme/Layout";
import BackToTopButton from "@theme/BackToTopButton";

import styles from "./navigation.module.css";
import { getNavigationData } from "./content";
import { sectionsMeta } from "./config";
import { Card } from "./components";
import { Hero } from "../components/Hero";

function Section({ section, content }) {
  const gridRef = useRef(null);

  useEffect(() => {
    const updateGridColumns = () => {
      const grid = gridRef.current;
      if (!grid) return;

      const containerWidth = grid.offsetWidth;
      const minCardWidth = 280;
      const gapWidth = 32;
      const maxColumns = 4;

      if (containerWidth < minCardWidth + gapWidth) {
        grid.style.gridTemplateColumns = "1fr";
        return;
      }

      const calculatedColumns = Math.min(
        Math.floor(containerWidth / (minCardWidth + gapWidth)),
        maxColumns
      );
      const columns = Math.max(calculatedColumns, 1);

      grid.style.gridTemplateColumns = `repeat(${columns}, minmax(${minCardWidth}px, 1fr))`;
    };

    updateGridColumns();
    window.addEventListener("resize", updateGridColumns);
    return () => window.removeEventListener("resize", updateGridColumns);
  }, []);

  return (
    <section id={section.id} className="padding-top--lg">
      <h2 className="text--center margin-bottom--md">{section.title}</h2>
      <hr className="margin-bottom--lg" />
      <div ref={gridRef} className={styles.cardGrid}>
        {content.map((item, itemIndex) => (
          <Card
            key={`${section.id}-${itemIndex}`}
            item={item}
            type={section.contentType}
          />
        ))}
      </div>
    </section>
  );
}

function NavigationContent({ sectionContents }) {
  return (
    <div className={styles.navigationContainer}>
      {sectionsMeta.map((section) => (
        <Section
          key={section.id}
          section={section}
          content={sectionContents[section.id]}
        />
      ))}
    </div>
  );
}

export function Navigation() {
  const sectionContents = getNavigationData();

  return (
    <Layout title="導航" description="最近的文章和專案">
      <BrowserOnly>
        {() => (
          <>
            <Hero
              title="導航"
              tagline="最近的文章和專案"
              className="hero--primary"
            />
            <main>
              <NavigationContent sectionContents={sectionContents} />
            </main>
            <BackToTopButton />
          </>
        )}
      </BrowserOnly>
    </Layout>
  );
}
