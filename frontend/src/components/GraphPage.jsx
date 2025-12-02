import React, { useState, useEffect, useRef, useMemo } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import ForceGraph2D from 'react-force-graph-2d';
import ShinyText from './ShinyText';

const GraphPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { kgResult, step1Output } = location.state || {};
  const [graphDimensions, setGraphDimensions] = useState({ width: 800, height: 600 });
  const [isPaused, setIsPaused] = useState(false);
  const [currentClusterIndex, setCurrentClusterIndex] = useState(0);
  const [graphReady, setGraphReady] = useState(false);
  const [isHypothesisLoading, setIsHypothesisLoading] = useState(false);
  const [isHypothesisReady, setIsHypothesisReady] = useState(false);
  const [hypothesisResult, setHypothesisResult] = useState(null);
  const hasHypothesisProcessed = useRef(false);
  const graphContainerRef = useRef(null);
  const graphRef = useRef(null);
  
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001';

  useEffect(() => {
    const updateDimensions = () => {
      if (graphContainerRef.current) {
        setGraphDimensions({
          width: graphContainerRef.current.clientWidth,
          height: graphContainerRef.current.clientHeight,
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // Call HypothesisAgent when page loads
  useEffect(() => {
    if (hasHypothesisProcessed.current) {
      return;
    }

    if (!kgResult || !kgResult.result || !step1Output) {
      return;
    }

    hasHypothesisProcessed.current = true;

    const generateHypothesis = async () => {
      setIsHypothesisLoading(true);
      try {
        const response = await fetch(`${API_URL}/api/generate-hypothesis`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            step1_output: step1Output,
            knowledge_graph: kgResult.result,
          }),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Failed to generate hypothesis');
        }

        const data = await response.json();
        setHypothesisResult(data.result);
        setIsHypothesisReady(true);
      } catch (err) {
        console.error('Error generating hypothesis:', err);
        // Don't set error state, just log it - hypothesis is optional
      } finally {
        setIsHypothesisLoading(false);
      }
    };

    generateHypothesis();
  }, [kgResult, step1Output, API_URL]);


  if (!kgResult || !kgResult.result) {
    return (
      <div className="min-h-screen w-full bg-black flex items-center justify-center">
        <div className="text-white text-xl">No graph data available</div>
      </div>
    );
  }

  // Prepare graph data - start with nodes from backend
  const initialNodes = kgResult.result.nodes.map((node, index) => ({
    id: typeof node === 'string' ? node : (node.id || node.label || `node-${index}`),
    name: typeof node === 'string' ? node : (node.label || node.id || `Node ${index}`),
  }));

  // Function to find matching node ID for an edge reference
  // Handles cases where edge reference is a prefix/suffix/substring of node name
  const findMatchingNodeId = (edgeRef, nodeMap) => {
    const normalizedRef = String(edgeRef || '').trim();
    if (!normalizedRef) return null;

    // Try exact match first
    if (nodeMap.has(normalizedRef)) {
      return normalizedRef;
    }

    // Try prefix matching: check if edge reference starts with a node name
    for (const [nodeId, node] of nodeMap.entries()) {
      if (normalizedRef.startsWith(nodeId) || nodeId.startsWith(normalizedRef)) {
        return nodeId;
      }
    }

    // Try substring matching: check if edge reference contains a node name or vice versa
    for (const [nodeId, node] of nodeMap.entries()) {
      if (normalizedRef.includes(nodeId) || nodeId.includes(normalizedRef)) {
        // Prefer longer matches (more specific)
        return nodeId;
      }
    }

    return null;
  };

  // Create a map of existing nodes (by ID and by name for lookup)
  const existingNodesMap = new Map(initialNodes.map(node => [node.id, node]));
  const nodeNamesMap = new Map(initialNodes.map(node => [node.name, node]));

  // Collect all node IDs referenced in edges and normalize them
  const referencedNodeIds = new Set();
  const edgeNodeMappings = new Map(); // Maps edge reference -> actual node ID

  kgResult.result.edges.forEach((edge) => {
    const sourceRef = String(edge.source || edge.from || '').trim();
    const targetRef = String(edge.target || edge.to || '').trim();
    
    if (sourceRef) {
      // Try to find matching node
      const sourceNodeId = findMatchingNodeId(sourceRef, existingNodesMap) || 
                          findMatchingNodeId(sourceRef, nodeNamesMap);
      if (sourceNodeId) {
        edgeNodeMappings.set(sourceRef, sourceNodeId);
        referencedNodeIds.add(sourceNodeId);
      } else {
        // No match found, will need to add as new node
        referencedNodeIds.add(sourceRef);
      }
    }
    
    if (targetRef) {
      const targetNodeId = findMatchingNodeId(targetRef, existingNodesMap) || 
                          findMatchingNodeId(targetRef, nodeNamesMap);
      if (targetNodeId) {
        edgeNodeMappings.set(targetRef, targetNodeId);
        referencedNodeIds.add(targetNodeId);
      } else {
        referencedNodeIds.add(targetRef);
      }
    }
  });

  // Add missing nodes that are referenced in edges but not matched to existing nodes
  const missingNodes = [];
  referencedNodeIds.forEach(nodeId => {
    if (!existingNodesMap.has(nodeId) && !edgeNodeMappings.has(nodeId)) {
      missingNodes.push({
        id: nodeId,
        name: nodeId,
      });
    }
  });

  // Combine initial nodes with missing nodes
  const nodes = [...initialNodes, ...missingNodes];

  // Create a set of valid node IDs for quick lookup
  const validNodeIds = new Set(nodes.map(node => node.id));

  // Prepare links with normalized node references
  const links = kgResult.result.edges
    .map((edge) => {
      const sourceRef = String(edge.source || edge.from || '').trim();
      const targetRef = String(edge.target || edge.to || '').trim();
      
      // Get normalized node IDs (use mapping if available, otherwise use original)
      const sourceId = edgeNodeMappings.get(sourceRef) || sourceRef;
      const targetId = edgeNodeMappings.get(targetRef) || targetRef;
      
      // Include edges where both source and target exist
      if (sourceId && targetId && validNodeIds.has(sourceId) && validNodeIds.has(targetId)) {
        return {
          source: sourceId,
          target: targetId,
          relation: edge.relation || edge.label || '',
        };
      }
      return null;
    })
    .filter(link => link !== null);

  // Debug: log graph statistics
  if (edgeNodeMappings.size > 0) {
    console.log(`Normalized ${edgeNodeMappings.size} edge references to match existing nodes`);
  }
  if (missingNodes.length > 0) {
    console.log(`Auto-added ${missingNodes.length} missing nodes:`, missingNodes.map(n => n.id));
  }
  console.log(`Graph loaded: ${nodes.length} nodes (${initialNodes.length} from backend, ${missingNodes.length} auto-added), ${links.length} edges`);

  // Detect clusters using connected components
  const clusters = useMemo(() => {
    if (!nodes.length || !links.length) return [];
    
    // Build adjacency map
    const adjacencyMap = new Map();
    nodes.forEach(node => {
      adjacencyMap.set(node.id, new Set());
    });
    
    links.forEach(link => {
      const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
      const targetId = typeof link.target === 'string' ? link.target : link.target.id;
      if (adjacencyMap.has(sourceId) && adjacencyMap.has(targetId)) {
        adjacencyMap.get(sourceId).add(targetId);
        adjacencyMap.get(targetId).add(sourceId);
      }
    });
    
    // Find connected components using DFS
    const visited = new Set();
    const clusters = [];
    
    const dfs = (nodeId, cluster) => {
      if (visited.has(nodeId)) return;
      visited.add(nodeId);
      cluster.push(nodeId);
      
      const neighbors = adjacencyMap.get(nodeId);
      if (neighbors) {
        neighbors.forEach(neighborId => {
          if (!visited.has(neighborId)) {
            dfs(neighborId, cluster);
          }
        });
      }
    };
    
    nodes.forEach(node => {
      if (!visited.has(node.id)) {
        const cluster = [];
        dfs(node.id, cluster);
        if (cluster.length > 0) {
          clusters.push(cluster);
        }
      }
    });
    
    return clusters;
  }, [nodes, links]);

  // Configure D3 forces for better edge length and node spacing
  useEffect(() => {
    if (graphRef.current && graphReady && clusters.length > 0) {
      try {
        const fg = graphRef.current;
        
        // Get D3 force instances
        const linkForce = fg.d3Force('link');
        const chargeForce = fg.d3Force('charge');
        
        if (linkForce) {
          // Set cluster-aware link distance if clusters are available
          if (clusters && clusters.length > 1) {
            linkForce.distance((link) => {
              try {
                // Get source and target IDs safely
                const sourceId = typeof link.source === 'object' ? (link.source?.id || link.source) : link.source;
                const targetId = typeof link.target === 'object' ? (link.target?.id || link.target) : link.target;
                
                if (!sourceId || !targetId) return 500;
                
                // Find which clusters contain these nodes
                let sourceClusterIndex = -1;
                let targetClusterIndex = -1;
                
                for (let i = 0; i < clusters.length; i++) {
                  if (clusters[i] && clusters[i].includes(sourceId)) {
                    sourceClusterIndex = i;
                  }
                  if (clusters[i] && clusters[i].includes(targetId)) {
                    targetClusterIndex = i;
                  }
                }
                
                // If nodes are in different clusters, use much larger distance
                if (sourceClusterIndex !== -1 && targetClusterIndex !== -1 && sourceClusterIndex !== targetClusterIndex) {
                  return 1500; // Much larger distance for inter-cluster edges
                }
                // Normal distance for edges within the same cluster
                return 250;
              } catch (e) {
                return 500; // Fallback
              }
            });
          } else {
            // Fallback: uniform distance if clusters not available
            linkForce.distance(() => 500);
          }
          // Reduce link strength to allow more flexibility
          linkForce.strength(0.2);
        }
        
        if (chargeForce) {
          // Set much stronger repulsion to spread clusters apart
          chargeForce.strength(-2500);
        }
        
        // Reheat simulation to apply new settings
        fg.d3ReheatSimulation();
      } catch (error) {
        console.error('Error configuring graph forces:', error);
        // Fallback: set basic forces if error occurs
        try {
          const fg = graphRef.current;
          const linkForce = fg.d3Force('link');
          const chargeForce = fg.d3Force('charge');
          if (linkForce) {
            linkForce.distance(() => 500);
            linkForce.strength(0.2);
          }
          if (chargeForce) {
            chargeForce.strength(-1200);
          }
          fg.d3ReheatSimulation();
        } catch (e) {
          console.error('Fallback force configuration failed:', e);
        }
      }
    }
  }, [graphReady, clusters]);

  // Focus on a cluster
  const focusOnCluster = (clusterIndex) => {
    if (!graphRef.current || !clusters[clusterIndex] || !graphReady) return;
    
    const cluster = clusters[clusterIndex];
    const clusterNodes = nodes.filter(node => cluster.includes(node.id));
    
    if (clusterNodes.length === 0) return;
    
    // Calculate bounding box of cluster
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let validNodes = 0;
    
    clusterNodes.forEach(node => {
      if (node.x !== undefined && node.y !== undefined && !isNaN(node.x) && !isNaN(node.y)) {
        minX = Math.min(minX, node.x);
        maxX = Math.max(maxX, node.x);
        minY = Math.min(minY, node.y);
        maxY = Math.max(maxY, node.y);
        validNodes++;
      }
    });
    
    if (validNodes === 0) return;
    
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const width = maxX - minX || 200; // Fallback if single node
    const height = maxY - minY || 200;
    
    // Calculate proper zoom level to fit cluster in viewport
    const viewWidth = graphDimensions.width || 800;
    const viewHeight = graphDimensions.height || 600;
    const padding = 10;
    const clusterWidth = width + padding * 2;
    const clusterHeight = height + padding * 2;
    
    // Calculate zoom level to fit cluster (smaller cluster = higher zoom)
    // We want the cluster to fill about 70% of the viewport
    const scaleX = (viewWidth * 0.7) / clusterWidth;
    const scaleY = (viewHeight * 0.7) / clusterHeight;
    const targetZoom = Math.min(scaleX, scaleY);
    
    // Center camera on cluster
    if (graphRef.current.centerAt) {
      graphRef.current.centerAt(centerX, centerY, 1000);
    }
    
    // Set zoom to calculated level (not multiply)
    setTimeout(() => {
      if (graphRef.current && graphRef.current.zoom) {
        // Set absolute zoom level, ensuring it's reasonable (between 0.5 and 5)
        const finalZoom = Math.max(0.5, Math.min(5, targetZoom));
        graphRef.current.zoom(finalZoom, 1000);
      }
    }, 300);
  };

  // Navigate to next/previous cluster
  const navigateCluster = (direction) => {
    if (clusters.length === 0) return;
    
    let newIndex;
    if (direction === 'next') {
      newIndex = (currentClusterIndex + 1) % clusters.length;
    } else {
      newIndex = (currentClusterIndex - 1 + clusters.length) % clusters.length;
    }
    
    setCurrentClusterIndex(newIndex);
    focusOnCluster(newIndex);
  };

  const containerWidth = graphDimensions.width || 800;
  const containerHeight = graphDimensions.height || 600;

  return (
    <div className="h-screen w-full bg-black relative flex flex-col overflow-hidden">
      {/* Navigation Arrows */}
      {clusters.length > 1 && graphReady && (
        <>
          <button
            onClick={() => navigateCluster('prev')}
            className="absolute top-1/2 transform -translate-y-1/2 z-20 text-white rounded-full p-10 transition-all hover:opacity-80"
            style={{ pointerEvents: 'auto', left: '60px' }}
          >
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M15 18l-6-6 6-6"/>
            </svg>
          </button>
          <button
            onClick={() => navigateCluster('next')}
            className="absolute top-1/2 transform -translate-y-1/2 z-20 text-white rounded-full p-10 transition-all hover:opacity-80"
            style={{ pointerEvents: 'auto', right: '60px' }}
          >
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M9 18l6-6-6-6"/>
            </svg>
          </button>
        </>
      )}
      {/* Big Window - Reduced height */}
      <div className="h-[87vh] w-full flex items-center justify-center px-8 pt-8 pb-4">
        <div className="w-full max-w-7xl h-full bg-black rounded-3xl p-8">
          <div 
            ref={graphContainerRef} 
            className="w-full h-full relative" 
            style={{ 
              height: 'calc(100% - 2rem)', 
              minHeight: '500px'
            }}
          >
            {/* Grid background layer */}
            <div 
              className="absolute inset-0 pointer-events-none"
              style={{
                backgroundImage: `
                  linear-gradient(to right, rgba(107, 114, 128, 0.4) 1px, transparent 1px),
                  linear-gradient(to bottom, rgba(107, 114, 128, 0.4) 1px, transparent 1px)
                `,
                backgroundSize: '50px 50px',
                backgroundPosition: '25px 25px',
                zIndex: 1
              }}
            />
            {/* Graph layer */}
            <div className="relative z-10 w-full h-full" style={{ backgroundColor: 'transparent' }}>
            {nodes.length === 0 || links.length === 0 ? (
              <div className="text-white p-4">No graph data available</div>
            ) : (
              <ForceGraph2D
                graphData={{ nodes, links }}
                nodeLabel={(node) => `${node.name}`}
                linkLabel={(link) => link.relation || ''}
                nodeColor={() => '#ffffff'}
                linkColor={() => '#94a3b8'}
                backgroundColor="transparent"
                width={containerWidth}
                height={containerHeight}
                nodeVal={(node) => Math.sqrt(node.name.length)}
                linkWidth={2}
                linkDirectionalArrowLength={0}
                linkDirectionalArrowRelPos={1}
                linkCanvasObject={(link, ctx, globalScale) => {
                  // Draw edge label with yellow background and text wrapping
                  if (link.relation) {
                    const start = typeof link.source === 'object' ? link.source : nodes.find(n => n.id === link.source);
                    const end = typeof link.target === 'object' ? link.target : nodes.find(n => n.id === link.target);
                    
                    if (start && end && start.x !== undefined && end.x !== undefined) {
                      const midX = (start.x + end.x) / 2;
                      const midY = (start.y + end.y) / 2;
                      
                      ctx.save();
                      const baseFontSize = 12;
                      const fontSize = baseFontSize / Math.max(globalScale, 0.5);
                      ctx.font = `${fontSize}px Courier`;
                      ctx.textAlign = 'center';
                      ctx.textBaseline = 'middle';
                      
                      // Wrap text to multiple lines if too long
                      const baseMaxWidth = 150; // Base max width at normal zoom
                      const maxWidth = baseMaxWidth * (fontSize / baseFontSize); // Scale with font size
                      
                      const words = link.relation.split(/\s+/);
                      const lines = [];
                      let currentLine = '';
                      
                      words.forEach((word) => {
                        const testLine = currentLine ? `${currentLine} ${word}` : word;
                        const metrics = ctx.measureText(testLine);
                        const testWidth = metrics.width;
                        
                        if (testWidth > maxWidth && currentLine) {
                          lines.push(currentLine);
                          currentLine = word;
                        } else {
                          currentLine = testLine;
                        }
                      });
                      
                      if (currentLine) {
                        lines.push(currentLine);
                      }
                      
                      // Calculate dimensions for background box
                      let maxLineWidth = 0;
                      lines.forEach(line => {
                        const metrics = ctx.measureText(line);
                        if (metrics.width > maxLineWidth) {
                          maxLineWidth = metrics.width;
                        }
                      });
                      
                      const padding = 4;
                      const lineHeight = fontSize * 1.2;
                      const totalHeight = lines.length * lineHeight;
                      const bgWidth = maxLineWidth + (padding * 2);
                      const bgHeight = totalHeight + (padding * 2);
                      
                      // Draw yellow background rectangle
                      ctx.fillStyle = '#fbbf24'; // Yellow color
                      ctx.fillRect(
                        midX - bgWidth / 2,
                        midY - bgHeight / 2,
                        bgWidth,
                        bgHeight
                      );
                      
                      // Draw text in black, wrapped across multiple lines
                      ctx.fillStyle = '#000000';
                      const startY = midY - (totalHeight / 2) + (lineHeight / 2);
                      lines.forEach((line, index) => {
                        ctx.fillText(line, midX, startY + (index * lineHeight));
                      });
                      
                      ctx.restore();
                    }
                  }
                }}
                linkCanvasObjectMode={() => 'after'}
                ref={graphRef}
                cooldownTicks={100}
                onEngineStop={() => {
                  // Graph has stabilized
                  setGraphReady(true);
                }}
                d3AlphaDecay={0.0228}
                d3VelocityDecay={0.4}
                linkStrength={0.3}
                linkCurvature={0}
                warmupTicks={0}
                onNodeDrag={(node) => {
                  // Allow free movement during drag
                }}
                onNodeDragEnd={(node) => {
                  // Fix node position after drag to keep it stable
                  node.fx = node.x;
                  node.fy = node.y;
                }}
                nodeCanvasObject={(node, ctx, globalScale) => {
                  const label = node.name;
                  const baseFontSize = 14;
                  const fontSize = baseFontSize / Math.max(globalScale, 0.5);
                  ctx.font = `${fontSize}px Courier`;
                  ctx.textAlign = 'center';
                  ctx.textBaseline = 'middle';
                  
                  // Wrap text to multiple lines if too long
                  // Use a consistent visual width - scale maxWidth with fontSize to maintain consistent wrapping
                  const baseMaxWidth = 120; // Base max width at normal zoom
                  const maxWidth = baseMaxWidth * (fontSize / baseFontSize); // Scale with font size
                  
                  const words = label.split(/\s+/);
                  const lines = [];
                  let currentLine = '';
                  
                  words.forEach((word) => {
                    const testLine = currentLine ? `${currentLine} ${word}` : word;
                    const metrics = ctx.measureText(testLine);
                    const testWidth = metrics.width;
                    
                    if (testWidth > maxWidth && currentLine) {
                        lines.push(currentLine);
                        currentLine = word;
                    } else {
                        currentLine = testLine;
                    }
                  });
                  
                  if (currentLine) {
                    lines.push(currentLine);
                  }
                  
                  // Calculate text dimensions
                  const lineHeight = fontSize * 1.2;
                  const totalHeight = lines.length * lineHeight;
                  const startY = node.y - (totalHeight / 2) + (lineHeight / 2);
                  
                  // Find the widest line for background width
                  let maxLineWidth = 0;
                  lines.forEach((line) => {
                    const metrics = ctx.measureText(line);
                    if (metrics.width > maxLineWidth) {
                      maxLineWidth = metrics.width;
                    }
                  });
                  
                  // Draw white background rectangle with rounded corners
                  const padding = 4;
                  const borderRadius = 6;
                  const bgWidth = maxLineWidth + (padding * 2);
                  const bgHeight = totalHeight + (padding * 2);
                  const bgX = node.x - (bgWidth / 2);
                  const bgY = node.y - (bgHeight / 2);
                  
                  ctx.fillStyle = '#ffffff';
                  ctx.beginPath();
                  ctx.roundRect(bgX, bgY, bgWidth, bgHeight, borderRadius);
                  ctx.fill();
                  
                  // Draw text in black
                  ctx.fillStyle = '#000000';
                  lines.forEach((line, index) => {
                    ctx.fillText(line, node.x, startY + (index * lineHeight));
                  });
                }}
                />
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Divider */}
      <div className="absolute bottom-[17vh] left-0 right-0 w-full flex justify-center px-8">
        <div className="w-full max-w-4xl border-t" style={{ borderColor: '#00A86B' }}></div>
      </div>

      {/* Status text below the window */}
      {isHypothesisLoading ? (
        <div className="absolute bottom-0 left-0 right-0 h-[15vh] w-full flex items-center justify-center">
          <ShinyText 
            text="Generating Hypothesis..." 
            disabled={false} 
            speed={3} 
            className='text-2xl' 
          />
        </div>
      ) : isHypothesisReady ? (
        <div className="absolute bottom-0 left-0 right-0 h-[15vh] w-full flex items-center justify-center">
          <div 
            className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={() => {
              if (isHypothesisReady && hypothesisResult && kgResult && step1Output) {
                navigate('/hypothesis', { 
                  state: { 
                    hypothesisResult,
                    step1Output,
                    knowledgeGraph: kgResult.result,
                    kgResult: kgResult
                  } 
                });
              }
            }}
          >
            <svg
              className="w-6 h-6 text-green-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M5 13l4 4L19 7"
              />
            </svg>
            <span className="text-2xl text-white animate-blink">
              Hypothesis Ready
            </span>
          </div>
        </div>
      ) : null}
    </div>
  );
};

export default GraphPage;

