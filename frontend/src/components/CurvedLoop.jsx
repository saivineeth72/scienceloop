import { useRef, useEffect, useState, useMemo, useId } from 'react';

const CurvedLoop = ({
  marqueeText = '',
  speed = 2,
  className,
  curveAmount = 400,
  direction = 'left',
  interactive = true,
  initialOffset = 0
}) => {
  const text = useMemo(() => {
    const hasTrailing = /\s|\u00A0$/.test(marqueeText);
    return (hasTrailing ? marqueeText.replace(/\s+$/, '') : marqueeText) + '\u00A0';
  }, [marqueeText]);
  const measureRef = useRef(null);
  const textPathRef = useRef(null);
  const pathRef = useRef(null);
  const [spacing, setSpacing] = useState(0);
  const [offset, setOffset] = useState(0);
  const uid = useId();
  const pathId = `curve-${uid}`;
  const pathD = `M-100,160 Q500,${160 + curveAmount} 1540,160`;

  const dragRef = useRef(false);
  const lastXRef = useRef(0);
  const dirRef = useRef(direction);
  const velRef = useRef(0);

  const textLength = spacing;
  // Generate enough text repetitions for seamless infinite loop
  // We need many repetitions so there's always text visible when wrapping
  const repetitions = textLength ? Math.max(50, Math.ceil(2000 / textLength)) : 1;
  const totalText = textLength
    ? Array(repetitions)
        .fill(text)
        .join('')
    : text;

  const ready = spacing > 0;

  useEffect(() => {
    if (measureRef.current) setSpacing(measureRef.current.getComputedTextLength());
  }, [text, className]);

  useEffect(() => {
    if (!spacing) return;
    if (textPathRef.current) {
      const initial = -spacing + initialOffset;
      textPathRef.current.setAttribute('startOffset', initial + 'px');
      setOffset(initial);
    }
  }, [spacing, initialOffset]);

  useEffect(() => {
    if (!spacing || !ready) return;
    let frame = 0;
    const step = () => {
      if (!dragRef.current && textPathRef.current) {
        const delta = dirRef.current === 'right' ? speed : -speed;
        const currentOffset = parseFloat(textPathRef.current.getAttribute('startOffset') || '0');
        let newOffset = currentOffset + delta;
        const wrapPoint = spacing;
        // Seamless wrapping - keep offset in range [-spacing, 0] for continuous loop
        if (newOffset <= -wrapPoint) {
          newOffset = newOffset + wrapPoint;
        }
        if (newOffset > 0) {
          newOffset = newOffset - wrapPoint;
        }
        textPathRef.current.setAttribute('startOffset', newOffset + 'px');
        setOffset(newOffset);
      }
      frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [spacing, speed, ready]);

  const onPointerDown = e => {
    if (!interactive) return;
    dragRef.current = true;
    lastXRef.current = e.clientX;
    velRef.current = 0;
    e.target.setPointerCapture(e.pointerId);
  };

  const onPointerMove = e => {
    if (!interactive || !dragRef.current || !textPathRef.current) return;
    const dx = e.clientX - lastXRef.current;
    lastXRef.current = e.clientX;
    velRef.current = dx;
    const currentOffset = parseFloat(textPathRef.current.getAttribute('startOffset') || '0');
    let newOffset = currentOffset + dx;
    const wrapPoint = spacing;
    // Seamless wrapping for drag interaction - keep in range [-spacing, 0]
    if (newOffset <= -wrapPoint) {
      newOffset = newOffset + wrapPoint;
    }
    if (newOffset > 0) {
      newOffset = newOffset - wrapPoint;
    }
    textPathRef.current.setAttribute('startOffset', newOffset + 'px');
    setOffset(newOffset);
  };

  const endDrag = () => {
    if (!interactive) return;
    dragRef.current = false;
    dirRef.current = velRef.current > 0 ? 'right' : 'left';
  };

  const cursorStyle = interactive ? (dragRef.current ? 'grabbing' : 'grab') : 'auto';

  return (
    <div
      className="w-full h-full flex items-center justify-center relative"
      style={{ visibility: ready ? 'visible' : 'hidden', cursor: cursorStyle }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerLeave={endDrag}
    >
      <div className="w-full relative">
        <svg
          className="select-none w-full block text-[4rem] md:text-[8rem] font-bold uppercase leading-none"
          viewBox="200 80 1240 160"
          preserveAspectRatio="xMidYMid meet"
          style={{ height: 'auto', display: 'block', overflow: 'visible' }}
        >
        <text ref={measureRef} xmlSpace="preserve" style={{ visibility: 'hidden', opacity: 0, pointerEvents: 'none' }}>
          {text}
        </text>
        <defs>
          <path ref={pathRef} id={pathId} d={pathD} fill="none" stroke="transparent" />
          <linearGradient id={`textGradient-${uid}`} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#000000" />
            <stop offset="25%" stopColor="#1a1a1a" />
            <stop offset="50%" stopColor="#333333" />
            <stop offset="75%" stopColor="#1a1a1a" />
            <stop offset="100%" stopColor="#000000" />
          </linearGradient>
        </defs>
        {ready && (
          <text xmlSpace="preserve" className={className ?? ''} fill={`url(#textGradient-${uid})`}>
            <textPath ref={textPathRef} href={`#${pathId}`} startOffset={offset + 'px'} xmlSpace="preserve">
              {totalText}
            </textPath>
          </text>
        )}
      </svg>
      </div>
    </div>
  );
};

export default CurvedLoop;

