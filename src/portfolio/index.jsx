import React, { useState, useEffect, useRef } from "react";
import BrowserOnly from "@docusaurus/BrowserOnly";
import Layout from "@theme/Layout";
import BackToTopButton from "@theme/BackToTopButton";

import styles from "./portfolio.module.css";
import { useDocsData } from "./content";
import { useSlideEffect, initFloatingElements } from "./animation";
import { getPortfolioContent } from "./content";
import { sectionsMeta } from "./config";
import { Card } from "./components";

function useFloatingElements(ref) {
  useEffect(() => {
    return initFloatingElements(ref, styles.heroBgElement);
  }, [ref]);
}

function HeroSection({ activeSection, switchSection, heroBackgroundRef }) {
  useFloatingElements(heroBackgroundRef);

  return (
    <div className={styles.hero}>
      <div className={styles.heroContent}>
        <h1 className={styles.mainTitle}>
          <span className={styles.titleHighlight}>導航</span>
        </h1>
        <h3 className={styles.subTitle}>最近的文章和專案</h3>
        <div className={styles.heroActions}>
          {sectionsMeta.map((section, index) => (
            <button
              key={section.id}
              className={`${styles.heroButton} ${
                activeSection === index ? styles.activeButton : ""
              }`}
              onClick={() => switchSection(index)}
            >
              {section.buttonText}
            </button>
          ))}
        </div>
      </div>
      <div ref={heroBackgroundRef} className={styles.heroBackground}></div>
    </div>
  );
}

function Section({ section, content, isActive, sectionRef }) {
  return (
    <div
      ref={sectionRef}
      className={`${styles.sectionWrapper} ${
        isActive ? styles.activeSectionWrapper : styles.inactiveSectionWrapper
      }`}
    >
      <section id={section.id} className={styles.section}>
        <h2 className={styles.sectionTitle}>{section.title}</h2>
        <div className={styles.grid}>
          {content.map((item, itemIndex) => (
            <Card key={`${section.id}-${itemIndex}`} item={item} type={section.contentType} />
          ))}
        </div>
      </section>
    </div>
  );
}

function PortfolioContent({ activeSection, isInitialized, isFirstRender, contentHeight, contentContainerRef, sectionWrapperRefs, sectionContents }) {
  return (
    <div className={styles.portfolioContainer}>
      <div
        ref={contentContainerRef}
        className={`${styles.contentContainer} ${
          !isFirstRender ? (activeSection === 0 ? styles.slideFromLeft : styles.slideFromRight) : styles.noTransition
        }`}
        style={{
          ...(!isInitialized ? { transition: "none" } : {}),
          overflow: "hidden",
          position: "relative",
          width: "100%",
          height: contentHeight,
        }}
      >
        {sectionsMeta.map((section, index) => (
          <Section
            key={section.id}
            section={section}
            content={sectionContents[section.id]}
            isActive={activeSection === index}
            sectionRef={(el) => (sectionWrapperRefs.current[index] = el)}
          />
        ))}
      </div>
    </div>
  );
}

export function Portfolio() {
  const docsMap = useDocsData();
  const heroBackgroundRef = useRef(null);
  const sectionContents = getPortfolioContent(docsMap);

  const {
    activeSection,
    isInitialized,
    isFirstRender,
    contentHeight,
    contentContainerRef,
    sectionWrapperRefs,
    switchSection,
  } = useSlideEffect(styles);

  return (
    <Layout title="導航" description="最近的文章和專案">
      <BrowserOnly>
        {() => (
          <>
            <HeroSection
              activeSection={activeSection}
              switchSection={switchSection}
              heroBackgroundRef={heroBackgroundRef}
            />
            <PortfolioContent
              activeSection={activeSection}
              isInitialized={isInitialized}
              isFirstRender={isFirstRender}
              contentHeight={contentHeight}
              contentContainerRef={contentContainerRef}
              sectionWrapperRefs={sectionWrapperRefs}
              sectionContents={sectionContents}
            />
            <BackToTopButton />
          </>
        )}
      </BrowserOnly>
    </Layout>
  );
}
