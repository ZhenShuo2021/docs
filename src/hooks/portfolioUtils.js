import { useState, useMemo, useEffect } from "react";
import { articles, projects } from "../data/data";

export const useDocsData = () => {
  const [docsMap, setDocsMap] = useState(new Map());

  useEffect(() => {
    const requireContext = require.context(
      "../../.docusaurus/docusaurus-plugin-content-docs",
      true,
      /site-docs-.*\.json$/
    );

    const loadAllDocs = async () => {
      const filePromises = requireContext.keys().map(async (file) => {
        try {
          const { permalink, frontMatter } = requireContext(file);
          return permalink ? [permalink, frontMatter] : null;
        } catch (error) {
          console.error(`無法讀取 ${file}:`, error);
          return null;
        }
      });

      const entries = await Promise.all(filePromises);
      const validEntries = entries.filter(Boolean);
      setDocsMap(new Map(validEntries));
    };

    loadAllDocs();
  }, []);

  return docsMap;
};

export const processFrontmatter = (item, docsMap) => {
  if (item.link?.startsWith("/")) {
    const frontmatter = docsMap.get(item.link) || {};
    return {
      ...item,
      title: item.title?.trim() || frontmatter.title || "找不到標題",
      description: item.description?.trim() || frontmatter.description || "",
      image: item.image?.trim() || frontmatter.image,
      tags: item.tags || frontmatter.tags || [""],
    };
  }

  return item;
};

export function getDisplayedContent(docsMap) {
  return useMemo(() => {
    return {
      displayedArticles: articles.map((item) => processFrontmatter(item, docsMap)),
      displayedProjects: projects.map((item) => processFrontmatter(item, docsMap)),
    };
  }, [docsMap]);
}

export function initFloatingElements(heroBackgroundRef, bgElementClass) {
  const heroBackground = heroBackgroundRef.current;
  if (!heroBackground) return;

  const elementCount = Math.floor(Math.random() * 4) + 4;
  const elements = [];

  class FloatingElement {
    constructor(element, allElements) {
      this.element = element;
      this.size = Math.random() * 150 + 80;
      this.speedX = (Math.random() * 0.3 + 0.1) * (Math.random() > 0.5 ? 1 : -1);
      this.speedY = (Math.random() * 0.3 + 0.1) * (Math.random() > 0.5 ? 1 : -1);
      this.mass = this.size;

      this.element.style.width = `${this.size}px`;
      this.element.style.height = `${this.size}px`;

      let attempts = 0;
      do {
        this.x = Math.random() * (heroBackground.clientWidth - this.size);
        this.y = Math.random() * (heroBackground.clientHeight - this.size);
        attempts++;
      } while (this.isOverlapping(allElements) && attempts < 100);

      this.element.style.left = `${this.x}px`;
      this.element.style.top = `${this.y}px`;
    }

    isOverlapping(allElements) {
      for (const other of allElements) {
        if (other === this) continue;
        const dx = this.x + this.size / 2 - (other.x + other.size / 2);
        const dy = this.y + this.size / 2 - (other.y + other.size / 2);
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance < (this.size + other.size) / 2) return true;
      }
      return false;
    }

    move() {
      let newX = this.x + this.speedX;
      let newY = this.y + this.speedY;

      if (newX <= 0 || newX + this.size >= heroBackground.clientWidth) {
        this.speedX = -this.speedX;
        newX = Math.max(0, Math.min(newX, heroBackground.clientWidth - this.size));
      }
      if (newY <= 0 || newY + this.size >= heroBackground.clientHeight) {
        this.speedY = -this.speedY;
        newY = Math.max(0, Math.min(newY, heroBackground.clientHeight - this.size));
      }

      this.x = newX;
      this.y = newY;
      this.element.style.left = `${newX}px`;
      this.element.style.top = `${newY}px`;
    }

    checkCollision(other) {
      const dx = other.x + other.size / 2 - (this.x + this.size / 2);
      const dy = other.y + other.size / 2 - (this.y + this.size / 2);
      const distance = Math.sqrt(dx * dx + dy * dy);
      const minDistance = (this.size + other.size) / 2;

      if (distance < minDistance) {
        const nx = dx / distance;
        const ny = dy / distance;

        const relativeVx = this.speedX - other.speedX;
        const relativeVy = this.speedY - other.speedY;

        const impulse = 2 * (relativeVx * nx + relativeVy * ny) / (this.mass + other.mass);

        this.speedX -= impulse * other.mass * nx;
        this.speedY -= impulse * other.mass * ny;
        other.speedX += impulse * this.mass * nx;
        other.speedY += impulse * this.mass * ny;

        const overlap = minDistance - distance;
        const correction = overlap / 2;
        this.x -= nx * correction;
        this.y -= ny * correction;
        other.x += nx * correction;
        other.y += ny * correction;

        this.x = Math.max(0, Math.min(this.x, heroBackground.clientWidth - this.size));
        this.y = Math.max(0, Math.min(this.y, heroBackground.clientHeight - this.size));
        other.x = Math.max(0, Math.min(other.x, heroBackground.clientWidth - other.size));
        other.y = Math.max(0, Math.min(other.y, heroBackground.clientHeight - other.size));

        this.element.style.left = `${this.x}px`;
        this.element.style.top = `${this.y}px`;
        other.element.style.left = `${other.x}px`;
        other.element.style.top = `${other.y}px`;
      }
    }
  }

  for (let i = 0; i < elementCount; i++) {
    const div = document.createElement("div");
    div.className = bgElementClass;
    heroBackground.appendChild(div);
    elements.push(new FloatingElement(div, elements));
  }

  let animationFrame;
  const animate = () => {
    elements.forEach((el) => el.move());
    for (let i = 0; i < elements.length; i++) {
      for (let j = i + 1; j < elements.length; j++) {
        elements[i].checkCollision(elements[j]);
      }
    }
    animationFrame = requestAnimationFrame(animate);
  };

  animationFrame = requestAnimationFrame(animate);

  return () => {
    cancelAnimationFrame(animationFrame);
  };
}
