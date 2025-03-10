import BrowserOnly from "@docusaurus/BrowserOnly";
import Layout from "@theme/Layout";
import BackToTopButton from "@theme/BackToTopButton";

import styles from "./portfolio.module.css";
import { useDocsData } from "./content";
import { getPortfolioContent } from "./content";
import { sectionsMeta } from "./config";
import { Card } from "./components";
import { Hero } from "../components/Hero";

function Section({ section, content }) {
  return (
    <div className={styles.sectionWrapper}>
      <section id={section.id} className={styles.section}>
        <h2 className={styles.sectionTitle}>{section.title}</h2>
        <hr></hr>
        <div className={styles.grid}>
          {content.map((item, itemIndex) => (
            <Card key={`${section.id}-${itemIndex}`} item={item} type={section.contentType} />
          ))}
        </div>
      </section>
    </div>
  );
}

function PortfolioContent({ sectionContents }) {
  return (
    <div className={styles.portfolioContainer}>
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

export function Portfolio() {
  const docsMap = useDocsData();
  const sectionContents = getPortfolioContent(docsMap);

  return (
    <Layout title="導航" description="最近的文章和專案">
      <BrowserOnly>
        {() => (
          <>
            <Hero title="導航" tagline="最近的文章和專案" className="custom-hero"/>
            <PortfolioContent sectionContents={sectionContents} />
            <BackToTopButton />
          </>
        )}
      </BrowserOnly>
    </Layout>
  );
}