import { useState, useEffect, useRef } from "react";

export function useSlideEffect(styles) {
  const [state, setState] = useState({
    activeSection: 0,
    isInitialized: false,
    isFirstRender: true,  // Add this to track first render
    contentHeight: "auto",
    slideDirection: "",
    touch: { isDragging: false, startX: 0, startY: 0, translateX: 0, isHorizontal: false, directionLocked: false },
    isMobile: false
  });
  
  const refs = {
    contentContainer: useRef(null),
    sectionWrappers: useRef([])
  };
  
  const updateState = (newState) => setState(prev => ({ ...prev, ...newState }));
  
  // Update content height with a more reliable approach
  const updateContentHeight = () => {
    if (refs.sectionWrappers.current[state.activeSection]) {
      const height = refs.sectionWrappers.current[state.activeSection].scrollHeight;
      updateState({ contentHeight: `${height}px` });
    }
  };

  // Initial setup
  useEffect(() => {
    const checkMobile = () => updateState({ isMobile: window.innerWidth <= 768 });
    checkMobile();
    window.addEventListener('resize', checkMobile);
    
    // Increase initial delay to ensure DOM is fully rendered
    const timer = setTimeout(() => {
      enableTransitions();
      updateContentHeight(); // Calculate height first
      positionSections(state.activeSection);
      updateState({ isInitialized: true, isFirstRender: false });
    }, 300); // Increased from 100ms to 300ms
    
    return () => {
      window.removeEventListener('resize', checkMobile);
      clearTimeout(timer);
    };
  }, []);
  
  // Recalculate height when active section changes
  useEffect(() => {
    if (state.isInitialized) {
      updateContentHeight();
    }
  }, [state.activeSection, state.isInitialized]);
  
  // Add window resize handler to update heights
  useEffect(() => {
    const handleResize = () => {
      if (state.isInitialized) {
        updateContentHeight();
      }
    };
    
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [state.isInitialized]);
  
  const totalSections = refs.sectionWrappers.current.length;
  
  useEffect(() => {
    const container = refs.contentContainer.current;
    if (!container || !state.isMobile) return;
    
    const handlers = {
      start: (e) => {
        const touch = e.touches[0];
        updateState({ 
          touch: { 
            ...state.touch, 
            isDragging: true, 
            startX: touch.clientX, 
            startY: touch.clientY, 
            translateX: 0, 
            isHorizontal: false, 
            directionLocked: false 
          } 
        });
        
        refs.sectionWrappers.current.forEach(ref => { if (ref) ref.style.transition = 'none'; });
      },
      
      move: (e) => {
        const { isDragging, startX, startY, directionLocked, isHorizontal } = state.touch;
        if (!isDragging || !totalSections) return;
        
        const touch = e.touches[0];
        const diffX = touch.clientX - startX;
        const diffY = touch.clientY - startY;
        
        if (!directionLocked) {
          const isHorizontalSwipe = Math.abs(diffX) > 25 && Math.abs(diffX) > Math.abs(diffY) * 1.5; // 水平閾值 25px
          const isVerticalSwipe = Math.abs(diffY) > 10; // 垂直閾值 10px
          
          if (isHorizontalSwipe || isVerticalSwipe) {
            updateState({ touch: { ...state.touch, isHorizontal: isHorizontalSwipe, directionLocked: true } });
            if (isHorizontalSwipe) e.preventDefault();
          }
        } else if (isHorizontal) {
          e.preventDefault();
          updateState({ touch: { ...state.touch, translateX: diffX } });
          
          refs.sectionWrappers.current.forEach((ref, index) => {
            if (!ref) return;
            
            ref.style.transform = index === state.activeSection ? `translateX(${diffX}px)` :
              (state.activeSection === 0 && index === totalSections - 1 && diffX > 0) ? `translateX(calc(-100% + ${diffX}px))` :
              (state.activeSection === totalSections - 1 && index === 0 && diffX < 0) ? `translateX(calc(100% + ${diffX}px))` :
              (index < state.activeSection) ? `translateX(calc(-100% + ${diffX}px))` : `translateX(calc(100% + ${diffX}px))`;
          });
        }
      },
      
      end: () => {
        const { isDragging, translateX, isHorizontal } = state.touch;
        if (!isDragging || !totalSections) return;
        
        updateState({ touch: { ...state.touch, isDragging: false } });
        enableTransitions();
        
        if (isHorizontal) {
          const threshold = 100; // 滑動閾值 100px
          let newIndex = state.activeSection;
          let newDirection = "";
          
          if (translateX > threshold) {
            newIndex = state.activeSection > 0 ? state.activeSection - 1 : totalSections - 1;
            newDirection = "left";
          } else if (translateX < -threshold) {
            newIndex = state.activeSection < totalSections - 1 ? state.activeSection + 1 : 0;
            newDirection = "right";
          }
          
          updateState({ activeSection: newIndex, slideDirection: newDirection });
          setTimeout(() => positionSections(newIndex), 50);
        } else {
          positionSections(state.activeSection);
        }
      }
    };
    
    container.addEventListener('touchstart', handlers.start, { passive: false });
    container.addEventListener('touchmove', handlers.move, { passive: false });
    container.addEventListener('touchend', handlers.end, { passive: false });
    
    return () => {
      container.removeEventListener('touchstart', handlers.start);
      container.removeEventListener('touchmove', handlers.move);
      container.removeEventListener('touchend', handlers.end);
    };
  }, [state, totalSections]);
  
  const enableTransitions = () => {
    if (refs.contentContainer.current) refs.contentContainer.current.style.transition = "";
    refs.sectionWrappers.current.forEach(ref => { if (ref) ref.style.transition = ""; });
  };
  
  const positionSections = (nextActiveIndex) => {
    refs.sectionWrappers.current.forEach((ref, index) => {
      if (!ref) return;
      
      const isActive = index === nextActiveIndex;
      Object.assign(ref.style, {
        transform: isActive ? "translateX(0)" : (index < nextActiveIndex ? "translateX(-100%)" : "translateX(100%)"),
        opacity: isActive ? "1" : "0",
        zIndex: isActive ? "10" : "1"
      });
    });
  };
  
  const switchSection = (index) => {
    if (index === state.activeSection || !totalSections) return;
    
    const direction = index < state.activeSection ? "left" : "right";
    const isEdgeCase = (state.activeSection === 0 && index === totalSections - 1) || 
                       (state.activeSection === totalSections - 1 && index === 0);
    
    requestAnimationFrame(() => {
      const container = refs.contentContainer.current;
      if (container) {
        container.style.transition = "none";
        container.classList.remove(styles.slideFromLeft, styles.slideFromRight);
        void container.offsetWidth;
        container.style.transition = "";
        
        container.classList.add(
          isEdgeCase 
            ? (state.activeSection === 0 ? styles.slideFromLeft : styles.slideFromRight)
            : (direction === "left" ? styles.slideFromLeft : styles.slideFromRight)
        );
      }
      
      positionSections(index);
      updateState({ activeSection: index, slideDirection: direction });
    });
  };
  
  return {
    activeSection: state.activeSection,
    isInitialized: state.isInitialized,
    isFirstRender: state.isFirstRender,  // Return this so it can be used in Portfolio component
    contentHeight: state.contentHeight,
    contentContainerRef: refs.contentContainer,
    sectionWrapperRefs: refs.sectionWrappers,
    switchSection,
    slideDirection: state.slideDirection,
    isDragging: state.touch.isDragging,
    translateX: state.touch.translateX
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
