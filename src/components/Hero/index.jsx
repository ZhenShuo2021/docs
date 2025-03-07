import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import styles from "./styles.module.css";

export function Hero() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header>
      <section className={styles.herowave}>
        <div className="content">
          <h2 className={styles.h2}>{siteConfig.title}</h2>
          <h3 className={styles.h3}>{siteConfig.tagline}</h3>
          <br />
        </div>
        <div className={styles.waves}></div>
      </section>
    </header>
  );
}
