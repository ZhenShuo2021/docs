import { useState, useEffect, useRef } from "react";

export function useSlideEffect(styles) {
  const [activeSection, setActiveSection] = useState(0);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isFirstRender, setIsFirstRender] = useState(true);
  const [contentHeight, setContentHeight] = useState("auto");
  const [slideDirection, setSlideDirection] = useState("");
  // 拖拽相關狀態
  const [isDragging, setIsDragging] = useState(false);
  const [startX, setStartX] = useState(0);
  const [translateX, setTranslateX] = useState(0);
  // 檢測是否為移動裝置
  const [isMobile, setIsMobile] = useState(false);

  const contentContainerRef = useRef(null);
  const sectionWrapperRefs = useRef([]);

  // 計算總頁數
  const totalSections = sectionWrapperRefs.current.length;

  // 檢測是否為移動裝置
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth <= 768);
    };
    
    checkMobile();
    window.addEventListener('resize', checkMobile);
    
    return () => {
      window.removeEventListener('resize', checkMobile);
    };
  }, []);

  useEffect(() => {
    const updateContentHeight = () => {
      if (sectionWrapperRefs.current[activeSection]) {
        setContentHeight(
          `${sectionWrapperRefs.current[activeSection].scrollHeight}px`
        );
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
        setIsInitialized(true);
      }, 100);
    }
  }, [isFirstRender]);

  // 啟用過渡動畫
  const enableTransitions = () => {
    if (contentContainerRef.current) {
      contentContainerRef.current.style.transition = "";
    }
    sectionWrapperRefs.current.forEach((ref) => {
      if (ref) ref.style.transition = "";
    });
  };

  // 處理滑動開始 - 只在移動裝置上啟用
  const handleDragStart = (e) => {
    if (!isMobile) return;
    
    const clientX = e.type === 'touchstart' ? e.touches[0].clientX : e.clientX;
    setIsDragging(true);
    setStartX(clientX);
    
    sectionWrapperRefs.current.forEach((ref) => {
      if (ref) ref.style.transition = 'none';
    });
  };

  // 處理滑動移動 - 只在移動裝置上啟用
  const handleDragMove = (e) => {
    if (!isMobile || !isDragging || !totalSections) return;
    
    const clientX = e.type === 'touchmove' ? e.touches[0].clientX : e.clientX;
    const diffX = clientX - startX;
    setTranslateX(diffX);

    sectionWrapperRefs.current.forEach((ref, index) => {
      if (!ref) return;
      
      if (index === activeSection) {
        ref.style.transform = `translateX(${diffX}px)`;
      } 
      // 處理邊緣情況的特殊效果
      else if (activeSection === 0 && index === totalSections - 1 && diffX > 0) {
        // 最左側往左滑，顯示最右側
        ref.style.transform = `translateX(calc(-100% + ${diffX}px))`;
      } 
      else if (activeSection === totalSections - 1 && index === 0 && diffX < 0) {
        // 最右側往右滑，顯示最左側
        ref.style.transform = `translateX(calc(100% + ${diffX}px))`;
      }
      else if (index < activeSection) {
        ref.style.transform = `translateX(calc(-100% + ${diffX}px))`;
      } 
      else {
        ref.style.transform = `translateX(calc(100% + ${diffX}px))`;
      }
    });
  };

  // 處理滑動結束 - 只在移動裝置上啟用
  const handleDragEnd = () => {
    if (!isMobile || !isDragging || !totalSections) return;
    
    setIsDragging(false);
    enableTransitions();

    const threshold = 50; // 滑動閾值
    let newIndex = activeSection;
    
    // 處理左右邊緣的循環滑動
    if (translateX > threshold) {
      // 向右滑動
      if (activeSection > 0) {
        newIndex = activeSection - 1;
      } else {
        // 最左側，循環到最右側
        newIndex = totalSections - 1;
      }
      setSlideDirection("left");
    } else if (translateX < -threshold) {
      // 向左滑動
      if (activeSection < totalSections - 1) {
        newIndex = activeSection + 1;
      } else {
        // 最右側，循環到最左側
        newIndex = 0;
      }
      setSlideDirection("right");
    }

    setActiveSection(newIndex);
    setTranslateX(0);
    
    // 延遲執行位置更新，確保動畫效果正確
    setTimeout(() => {
      positionSections(newIndex);
    }, 50);
  };

  // 定位各區塊
  const positionSections = (nextActiveIndex) => {
    sectionWrapperRefs.current.forEach((ref, index) => {
      if (!ref) return;
      
      if (index === nextActiveIndex) {
        ref.style.transform = "translateX(0)";
        ref.style.opacity = "1";
        ref.style.zIndex = "10";
      } else if (index < nextActiveIndex) {
        ref.style.transform = "translateX(-100%)";
        ref.style.opacity = "0";
        ref.style.zIndex = "1";
      } else {
        ref.style.transform = "translateX(100%)";
        ref.style.opacity = "0";
        ref.style.zIndex = "1";
      }
    });
  };

  const switchSection = (index) => {
    if (index === activeSection || !totalSections) return;

    const direction = index < activeSection ? "left" : "right";
    setSlideDirection(direction);

    // 處理邊緣情況的特殊效果
    let useSpecialEffect = false;
    if ((activeSection === 0 && index === totalSections - 1) || 
        (activeSection === totalSections - 1 && index === 0)) {
      useSpecialEffect = true;
    }

    requestAnimationFrame(() => {
      if (contentContainerRef.current) {
        contentContainerRef.current.style.transition = "none";
        contentContainerRef.current.classList.remove(
          styles.slideFromLeft,
          styles.slideFromRight
        );

        void contentContainerRef.current.offsetWidth;

        contentContainerRef.current.style.transition = "";
        
        // 處理循環滑動的特殊效果
        if (useSpecialEffect) {
          // 最左側到最右側 或 最右側到最左側
          if (activeSection === 0 && index === totalSections - 1) {
            contentContainerRef.current.classList.add(styles.slideFromLeft);
          } else {
            contentContainerRef.current.classList.add(styles.slideFromRight);
          }
        } else {
          contentContainerRef.current.classList.add(
            direction === "left" ? styles.slideFromLeft : styles.slideFromRight
          );
        }
      }

      positionSections(index);
      setActiveSection(index);
    });
  };

  useEffect(() => {
    if (isInitialized) {
      positionSections(activeSection);
    }
  }, [isInitialized]);

  // 添加事件監聽 - 只為移動裝置添加觸摸事件
  useEffect(() => {
    const container = contentContainerRef.current;
    if (!container) return;

    // 只為移動裝置添加觸摸事件
    container.addEventListener('touchstart', handleDragStart);
    container.addEventListener('touchmove', handleDragMove);
    container.addEventListener('touchend', handleDragEnd);

    return () => {
      container.removeEventListener('touchstart', handleDragStart);
      container.removeEventListener('touchmove', handleDragMove);
      container.removeEventListener('touchend', handleDragEnd);
    };
  }, [isDragging, startX, translateX, activeSection, totalSections, isMobile]);

  return {
    activeSection,
    isInitialized,
    isFirstRender,
    contentHeight,
    contentContainerRef,
    sectionWrapperRefs,
    switchSection,
    slideDirection,
    isDragging,
    translateX,
  };
}

export function initFloatingElements(heroBackgroundRef, bgElementClass) {
  const heroBackground = heroBackgroundRef.current;
  if (!heroBackground) return;

  const elementCount = Math.floor(Math.random() * 4) + 4;
  const elements = [];

  const heroArea = heroBackground.clientWidth * heroBackground.clientHeight;
  let currentOccupiedArea = 0;
  const minAreaPercentage = 0.2;
  const maxAreaPercentage = 0.25;
  const isMobile = window.innerWidth <= 768;

  class FloatingElement {
    constructor(element, allElements) {
      this.element = element;

      if (isMobile) {
        this.size = Math.random() * 80 + 60;
      } else {
        this.size = Math.random() * 150 + 80;
      }

      this.speedX =
        (Math.random() * 0.3 + 0.1) * (Math.random() > 0.5 ? 1 : -1);
      this.speedY =
        (Math.random() * 0.3 + 0.1) * (Math.random() > 0.5 ? 1 : -1);
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
        newX = Math.max(
          0,
          Math.min(newX, heroBackground.clientWidth - this.size)
        );
      }
      if (newY <= 0 || newY + this.size >= heroBackground.clientHeight) {
        this.speedY = -this.speedY;
        newY = Math.max(
          0,
          Math.min(newY, heroBackground.clientHeight - this.size)
        );
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

        const impulse =
          (2 * (relativeVx * nx + relativeVy * ny)) / (this.mass + other.mass);

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

        this.x = Math.max(
          0,
          Math.min(this.x, heroBackground.clientWidth - this.size)
        );
        this.y = Math.max(
          0,
          Math.min(this.y, heroBackground.clientHeight - this.size)
        );
        other.x = Math.max(
          0,
          Math.min(other.x, heroBackground.clientWidth - other.size)
        );
        other.y = Math.max(
          0,
          Math.min(other.y, heroBackground.clientHeight - other.size)
        );

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

    const tempElement = new FloatingElement(div, elements);
    const elementArea = Math.PI * Math.pow(tempElement.size / 2, 2);
    const newTotalArea = currentOccupiedArea + elementArea;
    const newAreaPercentage = newTotalArea / heroArea;

    if (
      newAreaPercentage <= maxAreaPercentage ||
      (i === elementCount - 1 && newAreaPercentage > maxAreaPercentage)
    ) {
      elements.push(tempElement);
      currentOccupiedArea = newTotalArea;
    } else {
      heroBackground.removeChild(div);
      break;
    }

    if (currentOccupiedArea / heroArea >= minAreaPercentage) {
      break;
    }
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
