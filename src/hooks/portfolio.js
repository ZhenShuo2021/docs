import { useState, useEffect, useRef } from 'react';

/**
 * 管理滑動效果的自定義 Hook
 * @param {Object} styles CSS 樣式對象
 * @returns {Object} 包含滑動效果相關的狀態和方法
 */
export function useSlideEffect(styles) {
  const [activeSection, setActiveSection] = useState("articles");  // projects or articles
  const [isInitialized, setIsInitialized] = useState(false);
  const [isFirstRender, setIsFirstRender] = useState(true);
  const [contentHeight, setContentHeight] = useState("auto");

  const contentContainerRef = useRef(null);
  const articlesWrapperRef = useRef(null);
  const projectsWrapperRef = useRef(null);

  // 動態計算並設置內容高度
  useEffect(() => {
    const updateContentHeight = () => {
      if (activeSection === "articles" && articlesWrapperRef.current) {
        setContentHeight(`${articlesWrapperRef.current.scrollHeight}px`);
      } else if (activeSection === "projects" && projectsWrapperRef.current) {
        setContentHeight(`${projectsWrapperRef.current.scrollHeight}px`);
      }
    };

    updateContentHeight();
    const timer = setTimeout(updateContentHeight, 100);
    return () => clearTimeout(timer);
  }, [activeSection]);

  useEffect(() => {
    if (isFirstRender) {
      setTimeout(() => {
        setIsFirstRender(false);
        enableTransitions();
      }, 100);
    }
  }, [isFirstRender, activeSection]);

  const enableTransitions = () => {
    if (contentContainerRef.current) {
      contentContainerRef.current.style.transition = '';
    }

    if (articlesWrapperRef.current) {
      articlesWrapperRef.current.style.transition = '';
    }

    if (projectsWrapperRef.current) {
      projectsWrapperRef.current.style.transition = '';
    }
  };

  const switchSection = (section) => {
    if (section === activeSection) return;

    requestAnimationFrame(() => {
      if (contentContainerRef.current) {
        contentContainerRef.current.style.transition = 'none';
        contentContainerRef.current.classList.remove(styles.slideFromLeft, styles.slideFromRight);

        void contentContainerRef.current.offsetWidth;

        contentContainerRef.current.style.transition = '';
        contentContainerRef.current.classList.add(
          section === "articles" ? styles.slideFromLeft : styles.slideFromRight
        );
      }

      setActiveSection(section);
    });
  };

  return {
    activeSection,
    isInitialized,
    isFirstRender,
    contentHeight,
    contentContainerRef,
    articlesWrapperRef,
    projectsWrapperRef,
    switchSection
  };
}
