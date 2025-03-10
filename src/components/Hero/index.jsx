import useDocusaurusContext from "@docusaurus/useDocusaurusContext";

export function Hero({ title, tagline, className }) {
  const { siteConfig } = useDocusaurusContext();
  const heroTitle = title || siteConfig.title;
  const heroTagline = tagline || siteConfig.tagline;

  return (
    <header className={className || styles.hero}>
      <section className={styles.herowave}>
        <div className="content">
          <h2 className={styles.h2}>{heroTitle}</h2>
          <h3 className={styles.h3}>{heroTagline}</h3>
          <br />
        </div>
        <div className="waves"></div>
      </section>
    </header>
  );
}
