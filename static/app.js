// Frontend logic for 2-stage Rubber Leaf AI Guard dashboard

document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const previewContainer = document.getElementById("previewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const bboxCanvas = document.getElementById("bboxCanvas");
    const removeBtn = document.getElementById("removeBtn");
    
    // Camera integration elements
    const startCameraBtn = document.getElementById("startCameraBtn");
    const cameraContainer = document.getElementById("cameraContainer");
    const videoFeed = document.getElementById("videoFeed");
    const captureBtn = document.getElementById("captureBtn");
    const stopCameraBtn = document.getElementById("stopCameraBtn");
    const cameraErrorMsg = document.getElementById("cameraErrorMsg");
    const videoWrapper = document.getElementById("videoWrapper");
    const toggleLiveDetectBtn = document.getElementById("toggleLiveDetectBtn");
    const videoCanvas = document.getElementById("videoCanvas");
    const cameraActionsPanel = document.getElementById("cameraActionsPanel");
    const toggleFullscreenBtn = document.getElementById("toggleFullscreenBtn");
    const saveBrowserSnapshotBtn = document.getElementById("saveBrowserSnapshotBtn");
    
    // Pi Physical Camera integration elements
    const startPiCameraBtn = document.getElementById("startPiCameraBtn");
    const stopPiCameraBtn = document.getElementById("stopPiCameraBtn");
    const capturePiBtn = document.getElementById("capturePiBtn");
    const savePiSnapshotBtn = document.getElementById("savePiSnapshotBtn");
    const piCameraContainer = document.getElementById("piCameraContainer");
    const piCameraStream = document.getElementById("piCameraStream");
    const togglePiAutoDetectBtn = document.getElementById("togglePiAutoDetectBtn");
    const togglePiAutoDetectBtnIcon = togglePiAutoDetectBtn ? togglePiAutoDetectBtn.querySelector("i") : null;
    
    // New Settings & Action Controls
    const boxThicknessSelect = document.getElementById("boxThicknessSelect");
    const showLabelsCheckbox = document.getElementById("showLabelsCheckbox");
    const exportCsvBtn = document.getElementById("exportCsvBtn");
    const printReportBtn = document.getElementById("printReportBtn");
    
    // Telemetry Elements
    const telemetryDetText = document.getElementById("telemetryDetText");
    const telemetryClsText = document.getElementById("telemetryClsText");
    const telemetryTotalText = document.getElementById("telemetryTotalText");
    const telemetryFpsText = document.getElementById("telemetryFpsText");
    const piCameraFpsText = document.getElementById("piCameraFpsText");
    
    let cameraStream = null;
    let isPiAutoDetecting = false;
    let sessionHistory = []; // Stored only in browser RAM
    let isLiveDetecting = false;
    let isProcessingFrame = false;
    let isFullscreen = false;
    let piStatusInterval = null;
    
    const resultsPlaceholder = document.getElementById("resultsPlaceholder");
    const resultsContent = document.getElementById("resultsContent");
    
    // Connection Status
    const systemIndicator = document.getElementById("system-status-indicator");
    const systemText = document.getElementById("system-status-text");
    
    // Stats elements
    const statPLFD = document.getElementById("stat-count-plfd");
    const statHealthy = document.getElementById("stat-count-healthy");
    const statOther = document.getElementById("stat-count-other");
    const statTotal = document.getElementById("stat-count-total");

    // Calibrated model detection threshold
    const DEFAULT_CONF = 0.15;

    let predictionData = null; // Store prediction results

    // --- INITIAL DATA LOAD & SYSTEM CHECKS ---
    async function loadStats() {
        try {
            const res = await fetch("/stats");
            const data = await res.json();
            
            // Populate stats safely if elements exist
            if (data && data.raw) {
                const plfdCount = data.raw["PLFD"] || 0;
                const healthyCount = data.raw["Healthy"] || 0;
                const otherCount = data.raw["Other Disease"] || 0;
                
                if (statPLFD) statPLFD.textContent = plfdCount;
                if (statHealthy) statHealthy.textContent = healthyCount;
                if (statOther) statOther.textContent = otherCount;
                if (statTotal) statTotal.textContent = plfdCount + healthyCount + otherCount;
            }
            
            // Check model status
            if (data && data.models_exist) {
                const stage1Ready = Boolean(data.models_exist.stage1);
                const stage2Ready = Boolean(data.models_exist.stage2);
                const isHailo = Boolean(data.hailo);
                
                if (stage1Ready && stage2Ready) {
                    if (systemIndicator) systemIndicator.className = "status-indicator ready";
                    if (systemText) systemText.textContent = isHailo ? "Models Ready (Hailo NPU)" : "Models Ready (2-Stage)";
                } else if (stage1Ready) {
                    if (systemIndicator) systemIndicator.className = "status-indicator training";
                    if (systemText) systemText.textContent = "Stage 1 Ready (Detector)";
                } else {
                    if (systemIndicator) systemIndicator.className = "status-indicator";
                    if (systemText) systemText.textContent = "Models Missing (Check weights)";
                }
            }
        } catch (e) {
            console.error("Error loading stats:", e);
            if (systemIndicator) systemIndicator.className = "status-indicator";
            if (systemText) systemText.textContent = "Server Connection Error";
        }
    }

    loadStats();

    // Dataset Refinement Pool Stats Loader
    async function loadDatasetStats() {
        try {
            const res = await fetch("/api/dataset/stats");
            const data = await res.json();
            const total = data.total || 0;
            const datasetStatusText = document.getElementById("dataset-status-text");
            if (datasetStatusText) {
                datasetStatusText.textContent = `Pool: ${total} samples`;
            }
            return data;
        } catch (e) {
            console.error("Error loading dataset stats:", e);
        }
    }

    loadDatasetStats();

    // --- GPS & FIELD GEOTAGGING SUBSYSTEM ---
    let currentClientCoords = null;
    let hardwareGpsLocked = false;

    function initGpsManager() {
        const gpsStatusText = document.getElementById("gps-status-text");
        const gpsStatusIcon = document.getElementById("gps-status-icon");

        // 1. Listen for smartphone/browser geolocation (fallback / mobile survey)
        if ("geolocation" in navigator) {
            navigator.geolocation.watchPosition(
                (pos) => {
                    currentClientCoords = {
                        latitude: pos.coords.latitude,
                        longitude: pos.coords.longitude,
                        accuracy: pos.coords.accuracy
                    };
                    if (!hardwareGpsLocked && gpsStatusText) {
                        gpsStatusText.textContent = `Mobile GPS: ${pos.coords.latitude.toFixed(4)}°, ${pos.coords.longitude.toFixed(4)}°`;
                        if (gpsStatusIcon) gpsStatusIcon.style.color = "#10b981";
                    }
                },
                (err) => {
                    console.log("[GPS] Browser geolocation notice:", err.message);
                },
                { enableHighAccuracy: true, maximumAge: 5000, timeout: 10000 }
            );
        }

        // 2. Poll Raspberry Pi hardware GNSS (/api/gps)
        async function pollHardwareGps() {
            try {
                const res = await fetch("/api/gps");
                const data = await res.json();
                if (data.status === "locked" && data.latitude && data.longitude) {
                    hardwareGpsLocked = true;
                    if (gpsStatusText) {
                        const satStr = data.satellites ? ` (${data.satellites} sats)` : "";
                        gpsStatusText.textContent = `NEO-M8N: ${data.latitude.toFixed(4)}°, ${data.longitude.toFixed(4)}°${satStr}`;
                    }
                    if (gpsStatusIcon) gpsStatusIcon.style.color = "#10b981";
                } else if (data.status === "searching") {
                    if (!currentClientCoords && gpsStatusText) {
                        gpsStatusText.textContent = "NEO-M8N: Searching satellites...";
                        if (gpsStatusIcon) gpsStatusIcon.style.color = "#f59e0b";
                    }
                } else {
                    if (!currentClientCoords && gpsStatusText) {
                        gpsStatusText.textContent = "GPS: Standby";
                        if (gpsStatusIcon) gpsStatusIcon.style.color = "var(--text-secondary)";
                    }
                }
            } catch (e) {
                // Ignore transient network errors
            }
        }

        pollHardwareGps();
        setInterval(pollHardwareGps, 3000);
    }

    initGpsManager();

    // --- PLANTATION DISEASE MAP & HEATMAP ENGINE ---
    let plantationMap = null;
    let mapMarkersLayer = null;
    let mapHeatLayer = null;
    let heatPoints = [];
    let mapStats = { plfd: 0, other: 0, healthy: 0 };
    let latestMapCenter = [7.1907, 125.4553]; // Default center (Mindanao, Philippines)

    function initPlantationMap() {
        const mapElem = document.getElementById("plantationMap");
        if (!mapElem || typeof L === "undefined") return;

        try {
            // Initialize Leaflet map
            plantationMap = L.map("plantationMap", {
                center: latestMapCenter,
                zoom: 15,
                zoomControl: true,
                attributionControl: false
            });

            // OpenStreetMap tile layer
            const tileLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
                maxZoom: 19,
                subdomains: ["a", "b", "c"]
            }).addTo(plantationMap);

            const offlineNotice = document.getElementById("mapOfflineNotice");
            tileLayer.on("tileerror", () => {
                if (offlineNotice) offlineNotice.style.display = "flex";
            });
            tileLayer.on("load", () => {
                if (offlineNotice) offlineNotice.style.display = "none";
            });

            // Create layers
            mapMarkersLayer = L.layerGroup().addTo(plantationMap);

            if (typeof L.heatLayer === "function") {
                mapHeatLayer = L.heatLayer([], {
                    radius: 35,
                    blur: 20,
                    maxZoom: 17,
                    gradient: {
                        0.2: "#10b981",  // Healthy (Green)
                        0.55: "#f59e0b", // Other disease (Amber)
                        0.85: "#ef4444"  // PLFD Hotspot (Red)
                    }
                }).addTo(plantationMap);
            }

            // Toggle Buttons
            const toggleHeatmapBtn = document.getElementById("toggleHeatmapBtn");
            const togglePinsBtn = document.getElementById("togglePinsBtn");
            const recenterMapBtn = document.getElementById("recenterMapBtn");
            const clearMapBtn = document.getElementById("clearMapBtn");

            if (toggleHeatmapBtn && mapHeatLayer) {
                toggleHeatmapBtn.addEventListener("click", () => {
                    if (plantationMap.hasLayer(mapHeatLayer)) {
                        plantationMap.removeLayer(mapHeatLayer);
                        toggleHeatmapBtn.classList.remove("active");
                        toggleHeatmapBtn.style.opacity = "0.5";
                    } else {
                        plantationMap.addLayer(mapHeatLayer);
                        toggleHeatmapBtn.classList.add("active");
                        toggleHeatmapBtn.style.opacity = "1";
                    }
                });
            }

            if (togglePinsBtn) {
                togglePinsBtn.addEventListener("click", () => {
                    if (plantationMap.hasLayer(mapMarkersLayer)) {
                        plantationMap.removeLayer(mapMarkersLayer);
                        togglePinsBtn.classList.remove("active");
                        togglePinsBtn.style.opacity = "0.5";
                    } else {
                        plantationMap.addLayer(mapMarkersLayer);
                        togglePinsBtn.classList.add("active");
                        togglePinsBtn.style.opacity = "1";
                    }
                });
            }

            if (recenterMapBtn) {
                recenterMapBtn.addEventListener("click", () => {
                    plantationMap.flyTo(latestMapCenter, 16, { duration: 1.2 });
                });
            }

            if (clearMapBtn) {
                clearMapBtn.addEventListener("click", () => {
                    mapMarkersLayer.clearLayers();
                    heatPoints = [];
                    if (mapHeatLayer) mapHeatLayer.setLatLngs([]);
                    mapStats = { plfd: 0, other: 0, healthy: 0 };
                    updateMapStatsUI();
                });
            }

            // Invalidate map size after DOM settles (fixes partial tile rendering)
            setTimeout(() => {
                plantationMap.invalidateSize();
            }, 600);
        } catch (e) {
            console.error("[Map] Leaflet initialization error:", e);
        }
    }

    function updateMapStatsUI() {
        const plfdElem = document.getElementById("mapStatPlfd");
        const otherElem = document.getElementById("mapStatOther");
        const healthyElem = document.getElementById("mapStatHealthy");

        if (plfdElem) plfdElem.textContent = `${mapStats.plfd} PLFD Hotspots`;
        if (otherElem) otherElem.textContent = `${mapStats.other} Other`;
        if (healthyElem) healthyElem.textContent = `${mapStats.healthy} Healthy`;
    }

    function plotDetectionsOnMap(locationData, leaves) {
        if (!plantationMap || !locationData || !locationData.latitude || !locationData.longitude) return;
        const lat = locationData.latitude;
        const lon = locationData.longitude;
        latestMapCenter = [lat, lon];

        let hasNewPoints = false;

        leaves.forEach((leaf) => {
            if (leaf.class_id === 0) return; // Skip ignored background leaves

            let color = "#10b981";
            let intensity = 0.25;

            if (leaf.final_diagnosis === "PLFD") {
                color = "#ef4444";
                intensity = 1.0;
                mapStats.plfd++;
            } else if (leaf.final_diagnosis === "Other Disease") {
                color = "#f59e0b";
                intensity = 0.6;
                mapStats.other++;
            } else {
                mapStats.healthy++;
            }

            hasNewPoints = true;
            // Add to heatmap data
            heatPoints.push([lat, lon, intensity]);

            // Add circle pin
            const marker = L.circleMarker([lat, lon], {
                radius: 8,
                fillColor: color,
                color: "#ffffff",
                weight: 2,
                opacity: 0.95,
                fillOpacity: 0.85
            });

            // Rich popup with thumbnail and prescription
            const popupContent = `
                <div style="font-family: sans-serif; font-size: 0.8rem; min-width: 170px; color: #1e293b;">
                    <div style="font-weight: 700; color: ${color}; margin-bottom: 4px; font-size: 0.85rem;">
                        ${leaf.final_diagnosis} (${(leaf.confidence * 100).toFixed(1)}%)
                    </div>
                    ${leaf.crop_path ? `<img src="${leaf.crop_path}" style="width: 100%; max-height: 90px; object-fit: cover; border-radius: 4px; margin-bottom: 4px;">` : ""}
                    <div style="color: #64748b; font-size: 0.72rem; line-height: 1.3;">
                        <strong>Coordinates:</strong> ${lat.toFixed(5)}°, ${lon.toFixed(5)}°<br>
                        <strong>GPS Source:</strong> ${locationData.source || "GPS"}<br>
                        <strong>Timestamp:</strong> ${new Date().toLocaleTimeString()}
                    </div>
                </div>
            `;
            marker.bindPopup(popupContent);
            mapMarkersLayer.addLayer(marker);
        });

        if (hasNewPoints) {
            if (mapHeatLayer) mapHeatLayer.setLatLngs(heatPoints);
            updateMapStatsUI();
            plantationMap.panTo([lat, lon]);
        }
    }

    initPlantationMap();

    // --- DRAW BOUNDING BOXES ON CANVAS ---
    function drawBoundingBoxes() {
        if (!predictionData || !predictionData.leaf_detected || !predictionData.leaves || predictionData.leaves.length === 0) {
            clearCanvas();
            return;
        }

        const ctx = bboxCanvas.getContext("2d");
        
        // Match canvas dimensions to the displayed size of the image
        const dispW = imagePreview.clientWidth;
        const dispH = imagePreview.clientHeight;
        
        bboxCanvas.width = dispW;
        bboxCanvas.height = dispH;
        
        // Clear canvas
        ctx.clearRect(0, 0, dispW, dispH);
        
        // Calculate scale factor from original image dimensions
        const natW = imagePreview.naturalWidth;
        const natH = imagePreview.naturalHeight;
        
        const scaleX = dispW / natW;
        const scaleY = dispH / natH;
        
        predictionData.leaves.forEach(leaf => {
            const box = leaf.box;
            
            // Box coordinates on displayed image
            const bx = box.x1 * scaleX;
            const by = box.y1 * scaleY;
            const bw = (box.x2 - box.x1) * scaleX;
            const bh = (box.y2 - box.y1) * scaleY;
            
            // Set colors based on diagnosis
            let color = '#f59e0b'; // Amber fallback
            const isIgnored = false; // Single-class model: all detections undergo Stage 2 disease classification
            
            if (isIgnored) {
                color = '#94a3b8'; // Slate grey for ignored
            } else if (leaf.final_diagnosis === "Healthy") {
                color = '#10b981'; // Green
            } else if (leaf.final_diagnosis === "PLFD") {
                color = '#ef4444'; // Red
            } else if (leaf.final_diagnosis === "Other Disease") {
                color = '#f59e0b'; // Orange/Amber
            }
            
            // Draw glow/shadow rect
            if (isIgnored) {
                ctx.shadowBlur = 0;
                ctx.setLineDash([4, 4]); // Dashed box for ignored
            } else {
                ctx.shadowColor = color;
                ctx.shadowBlur = 8;
                ctx.setLineDash([]); // Solid box
            }
            
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.5;
            ctx.strokeRect(bx, by, bw, bh);
            
            // Draw inner fill
            ctx.shadowBlur = 0; // reset shadow for fill
            ctx.setLineDash([]); // reset dashed for fill/text
            
            const rgbaFill = isIgnored ? 'rgba(148, 163, 184, 0.05)' :
                             color === '#10b981' ? 'rgba(16, 185, 129, 0.12)' : 
                             color === '#ef4444' ? 'rgba(239, 68, 68, 0.12)' : 'rgba(245, 158, 11, 0.12)';
            ctx.fillStyle = rgbaFill;
            ctx.fillRect(bx, by, bw, bh);
            
            // Draw label text background
            ctx.fillStyle = color;
            const labelText = isIgnored ? `Ignored Leaf (${(box.confidence * 100).toFixed(0)}%)` : `Leaf #${leaf.index} (${(box.confidence * 100).toFixed(0)}%)`;
            ctx.font = 'bold 11px sans-serif';
            const textWidth = ctx.measureText(labelText).width;
            ctx.fillRect(bx - 1.5, by - 16, textWidth + 8, 16);
            
            // Draw text
            ctx.fillStyle = '#ffffff';
            ctx.fillText(labelText, bx + 2.5, by - 4);
        });
    }

    function clearCanvas() {
        const ctx = bboxCanvas.getContext("2d");
        ctx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);
    }

    // Redraw bounding boxes on window resize (responsive scaling)
    window.addEventListener("resize", () => {
        if (predictionData && predictionData.leaf_detected) {
            drawBoundingBoxes();
        }
    });
    
    // Redraw once image completes loading
    imagePreview.addEventListener("load", () => {
        if (predictionData && predictionData.leaf_detected) {
            drawBoundingBoxes();
        }
    });

    // --- CAMERA STREAM FUNCTIONALITIES ---
    
    // Stop the camera stream safely
    function stopCamera() {
        stopLiveDetection();
        exitFullscreenMode();
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
        videoFeed.srcObject = null;
    }

    function startPiStatusPolling() {
        if (piStatusInterval) clearInterval(piStatusInterval);
        piStatusInterval = setInterval(async () => {
            if (piCameraContainer && !piCameraContainer.classList.contains("hidden")) {
                try {
                    const response = await fetch("/pi_status");
                    const data = await response.json();
                    const fpsVal = data.fps || 0.0;
                    if (piCameraFpsText) piCameraFpsText.textContent = `${fpsVal.toFixed(1)} FPS`;
                    if (telemetryFpsText) telemetryFpsText.textContent = `${fpsVal.toFixed(1)} FPS`;
                } catch (e) {
                    console.error("Error polling Pi status:", e);
                }
            }
        }, 1000);
    }

    function stopPiStatusPolling() {
        if (piStatusInterval) {
            clearInterval(piStatusInterval);
            piStatusInterval = null;
        }
        if (piCameraFpsText) piCameraFpsText.textContent = "-- FPS";
        if (telemetryFpsText) telemetryFpsText.textContent = "-- FPS";
    }

    // Raspberry Pi Physical Camera Management & Auto-Reconnection Engine
    let isPiCameraActive = false;
    let piStreamRetryTimer = null;

    function buildPiStreamUrl() {
        const thickness = boxThicknessSelect ? boxThicknessSelect.value : 2;
        const hideLabels = showLabelsCheckbox ? !showLabelsCheckbox.checked : false;
        const confVal = DEFAULT_CONF;
        return `/pi_video_feed?live_detect=${isPiAutoDetecting}&conf=${confVal}&box_thickness=${thickness}&hide_labels=${hideLabels}&_t=${Date.now()}`;
    }

    if (piCameraStream) {
        piCameraStream.onerror = () => {
            if (!isPiCameraActive) return;
            console.warn("[Pi Camera] Stream connection dropped or waiting for hardware. Reconnecting in 1.5s...");
            if (piCameraFpsText) piCameraFpsText.textContent = "Connecting...";
            if (piStreamRetryTimer) clearTimeout(piStreamRetryTimer);
            piStreamRetryTimer = setTimeout(() => {
                if (isPiCameraActive && piCameraStream) {
                    piCameraStream.src = buildPiStreamUrl();
                }
            }, 1500);
        };
        piCameraStream.onload = () => {
            if (piStreamRetryTimer) {
                clearTimeout(piStreamRetryTimer);
                piStreamRetryTimer = null;
            }
        };
    }

    // Stop the Raspberry Pi camera stream
    function stopPiCamera() {
        isPiCameraActive = false;
        if (piStreamRetryTimer) {
            clearTimeout(piStreamRetryTimer);
            piStreamRetryTimer = null;
        }
        isPiAutoDetecting = false;
        if (togglePiAutoDetectBtn) {
            togglePiAutoDetectBtn.classList.remove("active");
            togglePiAutoDetectBtn.innerHTML = `<i data-lucide="play" style="margin-right: 0.5rem; vertical-align: middle;"></i> Start Auto-Detect`;
            lucide.createIcons();
        }
        if (piCameraStream) {
            piCameraStream.src = "";
        }
        if (piCameraContainer) {
            piCameraContainer.classList.add("hidden");
        }
        dropzone.classList.remove("hidden");
        if (cameraActionsPanel) cameraActionsPanel.classList.remove("hidden");
        
        stopPiStatusPolling();
        
        // Release the camera on the server side
        fetch("/close_pi_camera", { method: "POST" })
            .then(res => res.json())
            .then(data => console.log("Pi camera released on server:", data))
            .catch(e => console.error("Error releasing Pi camera:", e));
    }

    // Start the Raspberry Pi camera stream
    function startPiCamera() {
        // Stop browser camera first if running
        stopCamera();
        cameraContainer.classList.add("hidden");
        
        // Hide normal preview if open
        previewContainer.classList.add("hidden");
        
        // Reset auto detect UI
        isPiAutoDetecting = false;
        if (togglePiAutoDetectBtn) {
            togglePiAutoDetectBtn.classList.remove("active");
            togglePiAutoDetectBtn.innerHTML = `<i data-lucide="play" style="margin-right: 0.5rem; vertical-align: middle;"></i> Start Auto-Detect`;
            lucide.createIcons();
        }

        // Show Pi camera elements
        dropzone.classList.add("hidden");
        if (cameraActionsPanel) cameraActionsPanel.classList.add("hidden");
        piCameraContainer.classList.remove("hidden");
        
        isPiCameraActive = true;
        if (piStreamRetryTimer) {
            clearTimeout(piStreamRetryTimer);
            piStreamRetryTimer = null;
        }

        startPiStatusPolling();
        
        // Set the src to the MJPEG route with cache-busting timestamp
        piCameraStream.src = buildPiStreamUrl();
    }

    const togglePiFullscreenBtn = document.getElementById("togglePiFullscreenBtn");
    if (togglePiFullscreenBtn) {
        togglePiFullscreenBtn.addEventListener("click", () => {
            const wrapper = document.getElementById("piVideoWrapper");
            if (!wrapper) return;
            if (!document.fullscreenElement) {
                if (wrapper.requestFullscreen) {
                    wrapper.requestFullscreen();
                } else if (wrapper.webkitRequestFullscreen) {
                    wrapper.webkitRequestFullscreen();
                }
                togglePiFullscreenBtn.innerHTML = `<i data-lucide="minimize" style="margin-right: 0.5rem;"></i> Exit Fullscreen`;
            } else {
                if (document.exitFullscreen) {
                    document.exitFullscreen();
                } else if (document.webkitExitFullscreen) {
                    document.webkitExitFullscreen();
                }
                togglePiFullscreenBtn.innerHTML = `<i data-lucide="maximize" style="margin-right: 0.5rem;"></i> Fullscreen`;
            }
            lucide.createIcons();
        });
    }

    document.addEventListener("fullscreenchange", () => {
        if (!document.fullscreenElement && togglePiFullscreenBtn) {
            togglePiFullscreenBtn.innerHTML = `<i data-lucide="maximize" style="margin-right: 0.5rem;"></i> Fullscreen`;
            lucide.createIcons();
        }
    });

    // Lossless IMX708 12MP Sensor Zoom Cycle (1.0x Wide -> 1.5x Standard -> 2.0x Canopy -> 3.0x Far)
    const cyclePiZoomBtn = document.getElementById("cyclePiZoomBtn");
    const piZoomBtnText = document.getElementById("piZoomBtnText");
    const ZOOM_LEVELS = [1.0, 1.5, 2.0, 3.0];
    let currentZoomIdx = 0;

    if (cyclePiZoomBtn) {
        cyclePiZoomBtn.addEventListener("click", async () => {
            currentZoomIdx = (currentZoomIdx + 1) % ZOOM_LEVELS.length;
            const nextZoom = ZOOM_LEVELS[currentZoomIdx];
            if (piZoomBtnText) {
                piZoomBtnText.textContent = `Zoom: ${nextZoom.toFixed(1)}x`;
            }
            if (nextZoom > 1.0) {
                cyclePiZoomBtn.classList.add("active");
            } else {
                cyclePiZoomBtn.classList.remove("active");
            }
            try {
                await fetch("/api/camera/zoom", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ zoom: nextZoom })
                });
            } catch (e) {
                console.error("Failed to set sensor zoom:", e);
            }
        });
    }

    function togglePiAutoDetect() {
        if (!piCameraStream) return;
        isPiAutoDetecting = !isPiAutoDetecting;
        
        if (isPiAutoDetecting) {
            togglePiAutoDetectBtn.classList.add("active");
            togglePiAutoDetectBtn.innerHTML = `<i data-lucide="square" style="margin-right: 0.5rem; vertical-align: middle;"></i> Stop Auto-Detect`;
            lucide.createIcons();
            piCameraStream.src = buildPiStreamUrl();
        } else {
            togglePiAutoDetectBtn.classList.remove("active");
            togglePiAutoDetectBtn.innerHTML = `<i data-lucide="play" style="margin-right: 0.5rem; vertical-align: middle;"></i> Start Auto-Detect`;
            lucide.createIcons();
            piCameraStream.src = buildPiStreamUrl();
        }
    }

    // Function to reload the active Pi camera feed when configurations change
    function updatePiCameraFeedSource() {
        if (isPiCameraActive && piCameraContainer && !piCameraContainer.classList.contains("hidden") && piCameraStream) {
            piCameraStream.src = buildPiStreamUrl();
        }
    }

    // Start camera stream handler
    async function startCamera() {
        // Reset states
        cameraErrorMsg.classList.add("hidden");
        cameraErrorMsg.innerHTML = "";
        videoWrapper.classList.remove("hidden");
        captureBtn.disabled = false;

        // Check for secure context or mediaDevices support
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showCameraError(
                "Secure Context Required",
                "WebRTC camera access is restricted in insecure contexts (HTTP). To use your phone's camera, you must access this app over HTTPS or configure your browser's exceptions.",
                `<ul>
                    <li>Option A: Run Flask with <code>python app.py --ssl</code> and access <code>https://${window.location.hostname}:5000</code></li>
                    <li>Option B (Chrome Android): Enable <code>chrome://flags/#unsafely-treat-insecure-origin-as-secure</code> and add <code>http://${window.location.hostname}:5000</code></li>
                </ul>`
            );
            videoWrapper.classList.add("hidden");
            captureBtn.disabled = true;
            
            // Show container to display error
            dropzone.classList.add("hidden");
            if (cameraActionsPanel) cameraActionsPanel.classList.add("hidden");
            cameraContainer.classList.remove("hidden");
            return;
        }

        try {
            // Request back camera preferred (facingMode environment)
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: { ideal: "environment" },
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            });

            cameraStream = stream;
            videoFeed.srcObject = stream;
            
            // UI updates
            dropzone.classList.add("hidden");
            if (cameraActionsPanel) cameraActionsPanel.classList.add("hidden");
            previewContainer.classList.add("hidden");
            cameraContainer.classList.remove("hidden");
        } catch (err) {
            console.error("Camera acquisition failed:", err);
            let userMsg = "Could not initialize video feed. Please ensure camera permissions are granted.";
            if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
                userMsg = "Camera permission was denied. Please reset website permissions in your browser and try again.";
            } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
                userMsg = "No video input device/camera found on this system.";
            }
            showCameraError("Camera Error", userMsg, `Technical details: <code>${err.name} - ${err.message}</code>`);
            videoWrapper.classList.add("hidden");
            captureBtn.disabled = true;
            
            // Show container to display error
            dropzone.classList.add("hidden");
            cameraContainer.classList.remove("hidden");
        }
    }

    function showCameraError(title, msg, helpHtml = "") {
        cameraErrorMsg.innerHTML = `
            <h4><i data-lucide="alert-octagon"></i> ${title}</h4>
            <p>${msg}</p>
            ${helpHtml}
        `;
        lucide.createIcons();
        cameraErrorMsg.classList.remove("hidden");
    }

    // Bind change event to cameraInput (native mobile camera)
    const cameraInput = document.getElementById("cameraInput");
    if (cameraInput) {
        cameraInput.addEventListener("change", (e) => {
            if (cameraInput.files.length > 0) {
                handleFileSelect(cameraInput.files[0]);
            }
        });
    }

    // Bind event to start camera btn
    startCameraBtn.addEventListener("click", (e) => {
        e.stopPropagation(); // Prevent opening the file upload dialog
        e.preventDefault();
        startCamera();
    });

    // Bind cancel btn
    stopCameraBtn.addEventListener("click", () => {
        stopCamera();
        cameraContainer.classList.add("hidden");
        dropzone.classList.remove("hidden");
        if (cameraActionsPanel) cameraActionsPanel.classList.remove("hidden");
    });

    // Bind event to start Pi camera btn
    if (startPiCameraBtn) {
        startPiCameraBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            e.preventDefault();
            startPiCamera();
        });
    }

    // Bind event to close Pi camera btn
    if (stopPiCameraBtn) {
        stopPiCameraBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            e.preventDefault();
            stopPiCamera();
        });
    }

    // Bind event to toggle Pi auto detect
    if (togglePiAutoDetectBtn) {
        togglePiAutoDetectBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            e.preventDefault();
            togglePiAutoDetect();
        });
    }

    // Real-time source update when settings change
    if (boxThicknessSelect) {
        boxThicknessSelect.addEventListener("change", () => {
            updatePiCameraFeedSource();
        });
    }
    if (showLabelsCheckbox) {
        showLabelsCheckbox.addEventListener("change", () => {
            updatePiCameraFeedSource();
        });
    }

    // FPS is monitored via server polling for reliability

    // Bind event to capture from Pi camera
    if (capturePiBtn) {
        capturePiBtn.addEventListener("click", async (e) => {
            e.stopPropagation();
            e.preventDefault();
            
            // Show loading state in the analysis report card
            resultsPlaceholder.innerHTML = `
                <i data-lucide="loader-2" class="placeholder-icon animate-pulse" style="animation: spin 1.5s linear infinite;"></i>
                <p>Capturing and running 2-stage leaf detection & classification from Pi Camera...</p>
            `;
            lucide.createIcons();
            resultsPlaceholder.classList.remove("hidden");
            resultsContent.classList.add("hidden");

            const confVal = DEFAULT_CONF;
            const imgszSelect = document.getElementById("imgszSelect");
            const imgszVal = imgszSelect ? parseInt(imgszSelect.value) : 640;

            const formData = new FormData();
            formData.append("conf", confVal);
            formData.append("imgsz", imgszVal);
            if (currentClientCoords) {
                formData.append("latitude", currentClientCoords.latitude);
                formData.append("longitude", currentClientCoords.longitude);
            }

            try {
                const response = await fetch("/predict_pi_camera", {
                    method: "POST",
                    body: formData
                });
                const result = await response.json();

                if (result.error) {
                    resultsPlaceholder.innerHTML = `
                        <i data-lucide="alert-triangle" class="placeholder-icon" style="color: var(--diseased-color);"></i>
                        <p style="color: var(--diseased-color); font-weight: 500;">${result.error}</p>
                    `;
                    lucide.createIcons();
                    return;
                }

                // Show preview of the captured frame
                imagePreview.src = result.image_url;
                
                // Hide Pi camera container and show standard preview container
                piCameraStream.src = "";
                piCameraContainer.classList.add("hidden");
                previewContainer.classList.remove("hidden");

                predictionData = result;
                displayResults(result);
            } catch (err) {
                console.error("Pi capture prediction error:", err);
                resultsPlaceholder.innerHTML = `
                    <i data-lucide="x-circle" class="placeholder-icon" style="color: var(--diseased-color);"></i>
                    <p>Connection error during local Pi camera execution.</p>
                `;
                lucide.createIcons();
            }
        });
    }

    // ── Toast Notification & Camera Shutter Effect System ──
    function showToastNotification(message, type = "success") {
        const container = document.getElementById("cameraToastContainer") || document.body;
        const toast = document.createElement("div");
        toast.className = `camera-toast toast-${type}`;
        const iconName = type === "success" ? "check-circle-2" : "alert-triangle";
        toast.innerHTML = `<i data-lucide="${iconName}" style="color: ${type === 'success' ? 'var(--lbp-green)' : 'var(--status-plfd)'};"></i><span>${message}</span>`;
        container.appendChild(toast);
        lucide.createIcons();
        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateY(20px) scale(0.95)";
            setTimeout(() => {
                if (toast.parentNode) toast.parentNode.removeChild(toast);
            }, 350);
        }, 3200);
    }

    function triggerCameraFlash(wrapperId) {
        const wrapper = document.getElementById(wrapperId);
        if (wrapper) {
            wrapper.classList.remove("shutter-flash");
            void wrapper.offsetWidth;
            wrapper.classList.add("shutter-flash");
            setTimeout(() => wrapper.classList.remove("shutter-flash"), 400);
        }
    }

    // ── Normal Capture from Pi Camera -> Save / Download to this Device (IncuVision style) ──
    if (savePiSnapshotBtn) {
        savePiSnapshotBtn.addEventListener("click", async (e) => {
            e.stopPropagation();
            e.preventDefault();
            
            triggerCameraFlash("piVideoWrapper");
            const originalHtml = savePiSnapshotBtn.innerHTML;
            savePiSnapshotBtn.disabled = true;
            savePiSnapshotBtn.innerHTML = `<i data-lucide="loader-2" style="animation: spin 1s linear infinite; margin-right: 0.35rem;"></i> Saving...`;
            lucide.createIcons();

            try {
                const now = new Date();
                const tsSlug = now.toISOString().replace(/[-:T]/g, "").slice(0, 14);
                const snapshotFilename = `RubberLeaf_PiSnapshot_${tsSlug}.jpg`;

                // Fetch high-res snapshot blob from Pi
                const response = await fetch(`/api/camera/snapshot?download=1&_t=${Date.now()}`);
                if (!response.ok) throw new Error("Pi camera hardware returned error status: " + response.status);
                
                const blob = await response.blob();
                const blobUrl = URL.createObjectURL(blob);

                // Direct browser download onto this device
                const downloadLink = document.createElement("a");
                downloadLink.href = blobUrl;
                downloadLink.download = snapshotFilename;
                document.body.appendChild(downloadLink);
                downloadLink.click();
                document.body.removeChild(downloadLink);
                URL.revokeObjectURL(blobUrl);

                showToastNotification(`📸 Photo saved to your device: ${snapshotFilename}`, "success");
            } catch (err) {
                console.error("Save snapshot error:", err);
                showToastNotification("Failed to capture snapshot from Pi camera.", "error");
            } finally {
                setTimeout(() => {
                    savePiSnapshotBtn.disabled = false;
                    savePiSnapshotBtn.innerHTML = originalHtml;
                    lucide.createIcons();
                }, 400);
            }
        });
    }

    // ── Normal Capture from Browser Camera -> Save / Download to this Device (IncuVision style) ──
    if (saveBrowserSnapshotBtn) {
        saveBrowserSnapshotBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            e.preventDefault();
            if (!videoFeed || videoFeed.readyState < 2) return;

            triggerCameraFlash("videoWrapper");
            const offCanvas = document.createElement("canvas");
            offCanvas.width = videoFeed.videoWidth || 1280;
            offCanvas.height = videoFeed.videoHeight || 720;
            const offCtx = offCanvas.getContext("2d");
            offCtx.drawImage(videoFeed, 0, 0, offCanvas.width, offCanvas.height);

            const now = new Date();
            const tsSlug = now.toISOString().replace(/[-:T]/g, "").slice(0, 14);
            const snapshotFilename = `RubberLeaf_BrowserSnapshot_${tsSlug}.jpg`;

            const link = document.createElement("a");
            link.download = snapshotFilename;
            link.href = offCanvas.toDataURL("image/jpeg", 0.95);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            showToastNotification(`📸 Photo saved to your device: ${snapshotFilename}`, "success");
        });
    }

    if (toggleLiveDetectBtn) {
        toggleLiveDetectBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            e.preventDefault();
            if (isLiveDetecting) {
                stopLiveDetection();
            } else {
                startLiveDetection();
            }
        });
    }

    if (toggleFullscreenBtn) {
        toggleFullscreenBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            e.preventDefault();
            if (isFullscreen) {
                exitFullscreenMode();
            } else {
                enterFullscreenMode();
            }
        });
    }

    function enterFullscreenMode() {
        if (!cameraStream) return;
        isFullscreen = true;
        
        // Move video wrapper to body to break out of card/grid CSS containment
        document.body.appendChild(videoWrapper);
        
        videoWrapper.classList.add("fullscreen");
        cameraContainer.classList.add("fullscreen-active");
        toggleFullscreenBtn.innerHTML = `<i data-lucide="minimize" style="width:16px;height:16px;vertical-align:middle;"></i> Exit Fullscreen`;
        lucide.createIcons();
        clearVideoCanvas();
    }

    function exitFullscreenMode() {
        isFullscreen = false;
        if (videoWrapper) {
            videoWrapper.classList.remove("fullscreen");
            // Teleport back to original position inside cameraContainer (before controls)
            const cameraControls = cameraContainer.querySelector(".camera-controls");
            if (cameraControls) {
                cameraContainer.insertBefore(videoWrapper, cameraControls);
            } else {
                cameraContainer.appendChild(videoWrapper);
            }
        }
        if (cameraContainer) cameraContainer.classList.remove("fullscreen-active");
        if (toggleFullscreenBtn) {
            toggleFullscreenBtn.innerHTML = `<i data-lucide="maximize" style="width:16px;height:16px;vertical-align:middle;"></i> Fullscreen`;
            lucide.createIcons();
        }
        clearVideoCanvas();
    }

    // --- LIVE VIDEO DETECTION LOOP ---
    function startLiveDetection() {
        if (!cameraStream) return;
        isLiveDetecting = true;
        toggleLiveDetectBtn.classList.add("active");
        toggleLiveDetectBtn.innerHTML = `<i data-lucide="square" style="width:16px;height:16px;vertical-align:middle;"></i> Stop Auto-Detect`;
        lucide.createIcons();
        
        // Hide normal results placeholder since we will show live updates
        resultsPlaceholder.classList.add("hidden");
        resultsContent.classList.remove("hidden");
        
        runLiveDetection();
    }

    function stopLiveDetection() {
        isLiveDetecting = false;
        isProcessingFrame = false;
        if (toggleLiveDetectBtn) {
            toggleLiveDetectBtn.classList.remove("active");
            toggleLiveDetectBtn.innerHTML = `<i data-lucide="play" style="width:16px;height:16px;vertical-align:middle;"></i> Start Auto-Detect`;
            lucide.createIcons();
        }
        clearVideoCanvas();
    }

    function clearVideoCanvas() {
        if (videoCanvas) {
            const ctx = videoCanvas.getContext("2d");
            ctx.clearRect(0, 0, videoCanvas.width, videoCanvas.height);
        }
    }

    async function runLiveDetection() {
        if (!isLiveDetecting || !cameraStream) return;
        
        if (isProcessingFrame) {
            requestAnimationFrame(runLiveDetection);
            return;
        }
        
        isProcessingFrame = true;
        
        // Grab current video frame
        const width = videoFeed.videoWidth || 640;
        const height = videoFeed.videoHeight || 480;
        
        const offscreenCanvas = document.createElement("canvas");
        offscreenCanvas.width = width;
        offscreenCanvas.height = height;
        const offscreenCtx = offscreenCanvas.getContext("2d");
        offscreenCtx.drawImage(videoFeed, 0, 0, width, height);
        
        // Convert to blob and send to backend
        offscreenCanvas.toBlob(async (blob) => {
            if (!blob) {
                isProcessingFrame = false;
                if (isLiveDetecting) requestAnimationFrame(runLiveDetection);
                return;
            }
            
            const confVal = confThreshold ? parseFloat(confThreshold.value) : 0.40;
            const imgszSelect = document.getElementById("imgszSelect");
            const imgszVal = imgszSelect ? parseInt(imgszSelect.value) : 1024;
            
            const formData = new FormData();
            formData.append("file", blob, "live_frame.jpg");
            formData.append("conf", confVal);
            formData.append("imgsz", imgszVal);
            
            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    body: formData
                });
                const result = await response.json();
                
                if (isLiveDetecting && !result.error) {
                    drawLiveBoundingBoxes(result);
                    displayResults(result);
                }
            } catch (e) {
                console.error("Live detection frame error:", e);
            } finally {
                isProcessingFrame = false;
                if (isLiveDetecting) {
                    // Small delay of 100ms to prevent resource choking on local host
                    setTimeout(() => {
                        if (isLiveDetecting) requestAnimationFrame(runLiveDetection);
                    }, 100);
                }
            }
        }, "image/jpeg", 0.70); // Compress to 70% to save bandwidth
    }

    function drawLiveBoundingBoxes(data) {
        if (!videoCanvas || !videoFeed) return;
        
        const ctx = videoCanvas.getContext("2d");
        const dispW = videoFeed.clientWidth;
        const dispH = videoFeed.clientHeight;
        
        videoCanvas.width = dispW;
        videoCanvas.height = dispH;
        
        ctx.clearRect(0, 0, dispW, dispH);
        
        if (!data.leaf_detected || !data.leaves || data.leaves.length === 0) {
            return;
        }
        
        const natW = videoFeed.videoWidth || 640;
        const natH = videoFeed.videoHeight || 480;
        
        const scaleX = dispW / natW;
        const scaleY = dispH / natH;
        
        data.leaves.forEach(leaf => {
            const box = leaf.box;
            const bx = box.x1 * scaleX;
            const by = box.y1 * scaleY;
            const bw = (box.x2 - box.x1) * scaleX;
            const bh = (box.y2 - box.y1) * scaleY;
            
            let color = '#f59e0b';
            const isIgnored = false; // Single-class model: all detections undergo Stage 2 disease classification
            
            if (isIgnored) {
                color = '#94a3b8';
            } else if (leaf.final_diagnosis === "Healthy") {
                color = '#10b981';
            } else if (leaf.final_diagnosis === "PLFD") {
                color = '#ef4444';
            } else if (leaf.final_diagnosis === "Other Disease") {
                color = '#f59e0b';
            }
            
            if (isIgnored) {
                ctx.shadowBlur = 0;
                ctx.setLineDash([4, 4]);
            } else {
                ctx.shadowColor = color;
                ctx.shadowBlur = 6;
                ctx.setLineDash([]);
            }
            
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.5;
            ctx.strokeRect(bx, by, bw, bh);
            
            ctx.shadowBlur = 0;
            ctx.setLineDash([]);
            
            // Draw label
            ctx.fillStyle = color;
            const labelText = isIgnored ? `Ignored (${(box.confidence * 100).toFixed(0)}%)` : `${leaf.final_diagnosis} (${(leaf.confidence * 100).toFixed(0)}%)`;
            ctx.font = 'bold 11px sans-serif';
            const textWidth = ctx.measureText(labelText).width;
            ctx.fillRect(bx - 1.5, by - 16, textWidth + 8, 16);
            
            ctx.fillStyle = '#ffffff';
            ctx.fillText(labelText, bx + 2.5, by - 4);
        });
    }

    // Bind capture frame btn
    captureBtn.addEventListener("click", () => {
        if (!cameraStream) return;

        // Create canvas offscreen
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        
        // Match sizes
        canvas.width = videoFeed.videoWidth || 640;
        canvas.height = videoFeed.videoHeight || 480;

        // Draw image frame
        ctx.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);

        // Convert canvas to jpeg blob
        canvas.toBlob((blob) => {
            if (!blob) {
                alert("Error capturing camera frame.");
                return;
            }

            // Create a File object so the existing upload function behaves normally
            const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
            
            // Stop camera
            stopCamera();
            cameraContainer.classList.add("hidden");
            
            // Show preview image locally
            const objectURL = URL.createObjectURL(blob);
            imagePreview.src = objectURL;
            previewContainer.classList.remove("hidden");

            // Upload the file to the 2-stage inference pipeline
            uploadImage(file);
        }, "image/jpeg", 0.95);
    });

    // =========================================================================
    //  In-Dashboard Refinement Review Deck (Capture -> Save or Discard)
    // =========================================================================
    let currentReviewData = null; // { temp_filename, image_url, image_base64, source: 'pi'|'browser', blob: Blob }
    let selectedRefinementCategory = "stage1_leaf";

    const refinementDeck = document.getElementById("refinementDeck");
    const refinementPreviewImg = document.getElementById("refinementPreviewImg");
    const refinementChips = document.querySelectorAll(".refinement-chip");
    const refinementCategoryCounter = document.getElementById("refinementCategoryCounter");
    const capturePiReviewBtn = document.getElementById("capturePiReviewBtn");
    const captureReviewBtn = document.getElementById("captureReviewBtn");
    const saveRefinementBtn = document.getElementById("saveRefinementBtn");
    const discardRefinementBtn = document.getElementById("discardRefinementBtn");
    const diagnoseRefinementBtn = document.getElementById("diagnoseRefinementBtn");
    const refinementNextShotBtn = document.getElementById("refinementNextShotBtn");
    const refinementStatusAlert = document.getElementById("refinementStatusAlert");
    const refinementAlertTitle = document.getElementById("refinementAlertTitle");
    const refinementAlertDesc = document.getElementById("refinementAlertDesc");

    // Category chips click handling
    refinementChips.forEach(chip => {
        chip.addEventListener("click", () => {
            refinementChips.forEach(c => c.classList.remove("active"));
            chip.classList.add("active");
            selectedRefinementCategory = chip.dataset.category;
            const heading = chip.querySelector(".chip-heading") ? chip.querySelector(".chip-heading").textContent : selectedRefinementCategory;
            if (refinementCategoryCounter) {
                refinementCategoryCounter.textContent = `Selected: ${heading}`;
            }
        });
    });

    function openRefinementDeck(imageUrl, metaLabel) {
        // Hide active viewfinders & previous containers
        if (piCameraContainer) piCameraContainer.classList.add("hidden");
        if (cameraContainer) cameraContainer.classList.add("hidden");
        if (dropzone) dropzone.classList.add("hidden");
        if (cameraActionsPanel) cameraActionsPanel.classList.add("hidden");
        if (previewContainer) previewContainer.classList.add("hidden");
        
        // Reset status alert
        if (refinementStatusAlert) refinementStatusAlert.classList.add("hidden");
        
        if (refinementPreviewImg) refinementPreviewImg.src = imageUrl;
        
        const refinementMetaTime = document.getElementById("refinementMetaTime");
        if (refinementMetaTime) {
            const now = new Date();
            refinementMetaTime.textContent = `${metaLabel} • ${now.toLocaleTimeString()}`;
        }
        
        if (refinementDeck) refinementDeck.classList.remove("hidden");
        lucide.createIcons();
    }

    function returnToLiveCamera() {
        if (refinementDeck) refinementDeck.classList.add("hidden");
        
        if (currentReviewData && currentReviewData.source === "pi") {
            startPiCamera();
        } else if (currentReviewData && currentReviewData.source === "browser") {
            startCamera();
        } else {
            if (dropzone) dropzone.classList.remove("hidden");
            if (cameraActionsPanel) cameraActionsPanel.classList.remove("hidden");
        }
        currentReviewData = null;
    }

    // 1. Pi Camera Capture & Review
    if (capturePiReviewBtn) {
        capturePiReviewBtn.addEventListener("click", async (e) => {
            e.stopPropagation();
            e.preventDefault();
            
            const originalHtml = capturePiReviewBtn.innerHTML;
            capturePiReviewBtn.disabled = true;
            capturePiReviewBtn.innerHTML = `<i data-lucide="loader-2" style="animation: spin 1s linear infinite;"></i> Snapping...`;
            lucide.createIcons();
            
            try {
                const formData = new FormData();
                const res = await fetch("/api/camera/snapshot", { method: "POST", body: formData });
                const data = await res.json();
                
                if (data.error) {
                    alert(data.error);
                    return;
                }
                
                currentReviewData = {
                    temp_filename: data.temp_filename,
                    image_url: data.image_url,
                    source: "pi"
                };
                
                openRefinementDeck(data.image_url, "Pi Camera IMX708 PDAF");
            } catch (err) {
                console.error("Pi Camera Snapshot error:", err);
                alert("Error capturing snapshot from Pi camera.");
            } finally {
                capturePiReviewBtn.disabled = false;
                capturePiReviewBtn.innerHTML = originalHtml;
                lucide.createIcons();
            }
        });
    }

    // 2. Browser Camera Capture & Review
    if (captureReviewBtn) {
        captureReviewBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            e.preventDefault();
            if (!cameraStream) return;
            
            const canvas = document.createElement("canvas");
            canvas.width = videoFeed.videoWidth || 1280;
            canvas.height = videoFeed.videoHeight || 720;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
            
            const dataUrl = canvas.toDataURL("image/jpeg", 0.95);
            canvas.toBlob((blob) => {
                currentReviewData = {
                    image_base64: dataUrl,
                    image_url: dataUrl,
                    blob: blob,
                    source: "browser"
                };
                openRefinementDeck(dataUrl, "Browser Camera Frame");
            }, "image/jpeg", 0.95);
        });
    }

    // 3. Save Sample to Dataset Refinement Pool
    if (saveRefinementBtn) {
        saveRefinementBtn.addEventListener("click", async () => {
            if (!currentReviewData) return;
            
            const originalHtml = saveRefinementBtn.innerHTML;
            saveRefinementBtn.disabled = true;
            saveRefinementBtn.innerHTML = `<i data-lucide="loader-2" style="animation: spin 1s linear infinite;"></i> Saving...`;
            lucide.createIcons();
            
            try {
                const formData = new FormData();
                formData.append("category", selectedRefinementCategory);
                if (currentReviewData.temp_filename) {
                    formData.append("temp_filename", currentReviewData.temp_filename);
                }
                if (currentReviewData.image_base64) {
                    formData.append("image_base64", currentReviewData.image_base64);
                }
                
                const res = await fetch("/api/dataset/save_sample", {
                    method: "POST",
                    body: formData
                });
                const data = await res.json();
                
                if (data.error) {
                    alert(data.error);
                    return;
                }
                
                // Show confirmation toast
                if (refinementAlertTitle) refinementAlertTitle.textContent = `Saved as ${data.category}!`;
                if (refinementAlertDesc) {
                    refinementAlertDesc.textContent = `Logged into manifest. Refinement Pool now contains ${data.stats.total} samples.`;
                }
                if (refinementStatusAlert) refinementStatusAlert.classList.remove("hidden");
                
                // Update header status pill
                const datasetStatusText = document.getElementById("dataset-status-text");
                if (datasetStatusText && data.stats) {
                    datasetStatusText.textContent = `Pool: ${data.stats.total} samples`;
                }
                
                lucide.createIcons();
            } catch (err) {
                console.error("Save sample error:", err);
                alert("Error saving sample to dataset.");
            } finally {
                saveRefinementBtn.disabled = false;
                saveRefinementBtn.innerHTML = originalHtml;
                lucide.createIcons();
            }
        });
    }

    // 4. Discard Sample & Retake
    if (discardRefinementBtn) {
        discardRefinementBtn.addEventListener("click", async () => {
            if (currentReviewData && currentReviewData.temp_filename) {
                const formData = new FormData();
                formData.append("temp_filename", currentReviewData.temp_filename);
                fetch("/api/dataset/discard_sample", { method: "POST", body: formData }).catch(e => console.error(e));
            }
            returnToLiveCamera();
        });
    }

    // 5. Next Shot Button
    if (refinementNextShotBtn) {
        refinementNextShotBtn.addEventListener("click", () => {
            returnToLiveCamera();
        });
    }

    // 6. Diagnose with AI from Refinement Deck
    if (diagnoseRefinementBtn) {
        diagnoseRefinementBtn.addEventListener("click", async () => {
            if (!currentReviewData) return;
            
            const originalHtml = diagnoseRefinementBtn.innerHTML;
            diagnoseRefinementBtn.disabled = true;
            diagnoseRefinementBtn.innerHTML = `<i data-lucide="loader-2" style="animation: spin 1s linear infinite;"></i> Running AI...`;
            lucide.createIcons();
            
            try {
                // If it came from browser cam blob, create a File
                let file = null;
                if (currentReviewData.blob) {
                    file = new File([currentReviewData.blob], "snapshot_diagnose.jpg", { type: "image/jpeg" });
                    uploadImage(file);
                } else if (currentReviewData.temp_filename) {
                    // Fetch the temp file as a blob and submit to /predict
                    const fetchRes = await fetch(currentReviewData.image_url);
                    const blob = await fetchRes.blob();
                    file = new File([blob], currentReviewData.temp_filename, { type: "image/jpeg" });
                    uploadImage(file);
                }
            } catch (err) {
                console.error("Diagnosis error:", err);
            } finally {
                diagnoseRefinementBtn.disabled = false;
                diagnoseRefinementBtn.innerHTML = originalHtml;
                lucide.createIcons();
            }
        });
    }

    // --- UPLOAD HANDLERS ---
    dropzone.addEventListener("click", () => fileInput.click());
    
    ["dragenter", "dragover"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add("dragover");
        }, false);
    });
    
    ["dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove("dragover");
        }, false);
    });
    
    dropzone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    
    fileInput.addEventListener("change", (e) => {
        if (fileInput.files.length > 0) {
            handleFileSelect(fileInput.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please select a valid image file.");
            return;
        }
        
        // Stop any active camera feeds
        stopCamera();
        stopPiCamera();
        
        // Reset prediction cache
        predictionData = null;
        clearCanvas();
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropzone.classList.add("hidden");
            if (cameraActionsPanel) cameraActionsPanel.classList.add("hidden");
            previewContainer.classList.remove("hidden");
            
            // Upload to server for prediction
            uploadImage(file);
        };
        reader.readAsDataURL(file);
    }

    removeBtn.addEventListener("click", () => {
        // Reset preview
        imagePreview.src = "";
        previewContainer.classList.add("hidden");
        dropzone.classList.remove("hidden");
        if (cameraActionsPanel) cameraActionsPanel.classList.remove("hidden");
        fileInput.value = "";
        predictionData = null;
        clearCanvas();
        
        // Reset results UI
        resultsPlaceholder.classList.remove("hidden");
        resultsContent.classList.add("hidden");
    });

    // --- INFERENCE PIPELINE CLIENT ---
    async function uploadImage(file) {
        // Clear results, show loading
        resultsPlaceholder.innerHTML = `
            <i data-lucide="loader-2" class="placeholder-icon animate-pulse" style="animation: spin 1.5s linear infinite;"></i>
            <p>Running 2-stage leaf detection & classification...</p>
        `;
        lucide.createIcons();
        resultsPlaceholder.classList.remove("hidden");
        resultsContent.classList.add("hidden");

        const confVal = confThreshold ? parseFloat(confThreshold.value) : 0.40;
        const imgszSelect = document.getElementById("imgszSelect");
        const imgszVal = imgszSelect ? parseInt(imgszSelect.value) : 1024;

        const formData = new FormData();
        formData.append("file", file);
        formData.append("conf", confVal);
        formData.append("imgsz", imgszVal);
        if (currentClientCoords) {
            formData.append("latitude", currentClientCoords.latitude);
            formData.append("longitude", currentClientCoords.longitude);
        }

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });
            const result = await response.json();
            
            if (result.error) {
                resultsPlaceholder.innerHTML = `
                    <i data-lucide="alert-triangle" class="placeholder-icon" style="color: var(--diseased-color);"></i>
                    <p style="color: var(--diseased-color); font-weight: 500;">${result.error}</p>
                `;
                lucide.createIcons();
                return;
            }
            
            predictionData = result;
            displayResults(result);
        } catch (e) {
            console.error("Prediction error:", e);
            resultsPlaceholder.innerHTML = `
                <i data-lucide="x-circle" class="placeholder-icon" style="color: var(--diseased-color);"></i>
                <p>Connection error during model execution.</p>
            `;
            lucide.createIcons();
        }
    }

    function displayResults(data) {
        resultsPlaceholder.classList.add("hidden");
        resultsContent.classList.remove("hidden");
        
        // Draw all bounding boxes on canvas
        drawBoundingBoxes();
        
        // Render Telemetry
        if (data.telemetry) {
            if (telemetryDetText) telemetryDetText.textContent = `${data.telemetry.det_latency_ms} ms`;
            if (telemetryClsText) telemetryClsText.textContent = `${data.telemetry.cls_latency_ms} ms`;
            if (telemetryTotalText) {
                telemetryTotalText.textContent = `${data.telemetry.total_latency_ms} ms`;
                const ms = data.telemetry.total_latency_ms;
                if (ms < 250) {
                    telemetryTotalText.style.color = "var(--healthy-color)";
                } else if (ms < 600) {
                    telemetryTotalText.style.color = "#f59e0b";
                } else {
                    telemetryTotalText.style.color = "var(--diseased-color)";
                }
            }
        } else {
            if (telemetryDetText) telemetryDetText.textContent = "-- ms";
            if (telemetryClsText) telemetryClsText.textContent = "-- ms";
            if (telemetryTotalText) telemetryTotalText.textContent = "-- ms";
        }

        // Push to Session History array (in RAM)
        if (data.leaves && data.leaves.length > 0) {
            const timestamp = new Date().toLocaleString();
            data.leaves.forEach(leaf => {
                sessionHistory.push({
                    timestamp: timestamp,
                    filename: data.image_url.split('/').pop(),
                    index: leaf.index,
                    stage1_confidence: leaf.box ? leaf.box.confidence : (leaf.confidence || 0.0),
                    class_id: leaf.class_id,
                    diagnosis: leaf.final_diagnosis,
                    diagnosis_confidence: leaf.confidence || 0.0,
                    det_latency_ms: data.telemetry ? data.telemetry.det_latency_ms : 0.0,
                    cls_latency_ms: data.telemetry ? data.telemetry.cls_latency_ms : 0.0,
                    total_latency: data.telemetry ? data.telemetry.total_latency_ms : 0.0,
                    latitude: (data.location && data.location.latitude) ? data.location.latitude : "",
                    longitude: (data.location && data.location.longitude) ? data.location.longitude : "",
                    gps_source: (data.location && data.location.source) ? data.location.source : "",
                    maps_url: (data.location && data.location.maps_url) ? data.location.maps_url : ""
                });
            });
        }

        // Render Location Banner if available
        const reportLocationBanner = document.getElementById("reportLocationBanner");
        const reportLocationCoords = document.getElementById("reportLocationCoords");
        const reportLocationSource = document.getElementById("reportLocationSource");
        const reportLocationMapLink = document.getElementById("reportLocationMapLink");

        if (reportLocationBanner && data.location && data.location.latitude && data.location.longitude) {
            if (reportLocationCoords) {
                reportLocationCoords.textContent = `${data.location.latitude.toFixed(5)}°, ${data.location.longitude.toFixed(5)}°`;
            }
            if (reportLocationSource) {
                reportLocationSource.textContent = data.location.source || "GPS";
            }
            if (reportLocationMapLink) {
                reportLocationMapLink.href = data.location.maps_url || `https://www.google.com/maps?q=${data.location.latitude},${data.location.longitude}`;
            }
            reportLocationBanner.classList.remove("hidden");

            // Plot detections directly on the Plantation Surveillance Heatmap
            if (data.leaves && data.leaves.length > 0) {
                plotDetectionsOnMap(data.location, data.leaves);
            }
        } else if (reportLocationBanner) {
            reportLocationBanner.classList.add("hidden");
        }

        const summaryTotalText = document.getElementById("summaryTotalText");
        const summaryStatsBadges = document.getElementById("summaryStatsBadges");
        const multiResultsContainer = document.getElementById("multiResultsContainer");
        
        // Clear previous list
        multiResultsContainer.innerHTML = "";
        summaryStatsBadges.innerHTML = "";
        
        if (!data.leaf_detected || !data.leaves || data.leaves.length === 0) {
            summaryTotalText.textContent = "No Leaves Detected";
            summaryStatsBadges.innerHTML = `<span class="badge plfd">0 Leaf Matches</span>`;
            
            multiResultsContainer.innerHTML = `
                <div class="leaf-result-card" style="border-left: 4px solid var(--diseased-color);">
                    <div class="leaf-card-header">
                        <span class="leaf-index-badge">No Leaf Found</span>
                    </div>
                    <div class="leaf-card-body" style="grid-template-columns: 1fr; padding: 1.5rem; text-align: center; color: var(--text-secondary);">
                        <p>Stage 1 was unable to identify any rubber tree leaves in the image. Please verify you are using a clear leaf shot under normal lighting.</p>
                    </div>
                </div>
            `;
            return;
        }
        
        // Count totals
        const totalCount = data.leaves.length;
        let healthyCount = 0;
        let plfdCount = 0;
        let otherCount = 0;
        let ignoredCount = 0;
        
        data.leaves.forEach(leaf => {
            if (leaf.class_id === 0) {
                ignoredCount++;
            } else if (leaf.final_diagnosis === "Healthy") {
                healthyCount++;
            } else if (leaf.final_diagnosis === "PLFD") {
                plfdCount++;
            } else if (leaf.final_diagnosis === "Other Disease") {
                otherCount++;
            }
        });
        
        const activeCount = totalCount - ignoredCount;
        summaryTotalText.textContent = `${activeCount} Leaf${activeCount !== 1 ? 'ves' : ''} Classified`;
        if (ignoredCount > 0) {
            summaryTotalText.textContent += ` (${ignoredCount} Ignored)`;
        }
        
        // Add badges
        if (healthyCount > 0) {
            summaryStatsBadges.innerHTML += `<span class="badge healthy">${healthyCount} Healthy</span>`;
        }
        if (otherCount > 0) {
            summaryStatsBadges.innerHTML += `<span class="badge warning">${otherCount} Other</span>`;
        }
        if (plfdCount > 0) {
            summaryStatsBadges.innerHTML += `<span class="badge plfd">${plfdCount} PLFD</span>`;
        }
        if (ignoredCount > 0) {
            summaryStatsBadges.innerHTML += `<span class="badge ignored">${ignoredCount} Ignored</span>`;
        }
        
        // Populate dynamic leaf diagnosis cards
        data.leaves.forEach(leaf => {
            if (leaf.class_id === 0) {
                // Render ignored leaf card
                const cardHtml = `
                    <div class="leaf-result-card ignored">
                        <div class="leaf-card-header">
                            <span class="leaf-index-badge">Ignored Object #${leaf.index}</span>
                            <span class="badge ignored">IGNORED (${(leaf.box.confidence * 100).toFixed(0)}%)</span>
                        </div>
                        <div class="leaf-card-body" style="grid-template-columns: 1fr; padding: 0.8rem 1rem; color: var(--text-secondary); font-size: 0.8rem; line-height: 1.4;">
                            <p><i data-lucide="info" style="display: inline-block; vertical-align: middle; width: 14px; height: 14px; margin-right: 0.25rem;"></i> Classified as <strong>Ignore Leaf</strong> (Class 0) by Stage 1 detector. Bypassed Stage 2 classification to conserve processing resources.</p>
                        </div>
                    </div>
                `;
                multiResultsContainer.insertAdjacentHTML("beforeend", cardHtml);
                return;
            }
            const healthyVal = leaf.probs["Healthy"] || 0;
            const otherVal = leaf.probs["Other Disease"] || 0;
            const plfdVal = leaf.probs["PLFD"] || 0;
            
            // Set styles and text based on diagnosis
            let cardClass = "";
            let diagnosisText = leaf.final_diagnosis.toUpperCase();
            let recIconText = "💡";
            let recTitleText = "Recommendation";
            let recBodyText = "";
            let colorBorder = "var(--border-medium)";
            
            if (leaf.final_diagnosis === "Healthy") {
                cardClass = "healthy";
                recIconText = "🛡️";
                recTitleText = "Leaf Health Secured";
                recBodyText = "No signs of Pestalotiopsis Leaf Fall Disease detected. Maintain standard monitoring schedules and good soil drainage.";
                colorBorder = "var(--status-healthy)";
            } else if (leaf.final_diagnosis === "PLFD") {
                cardClass = "plfd";
                recIconText = "⚠️";
                recTitleText = "Action Required: PLFD Detected";
                recBodyText = "PLFD identified! Apply systemic fungicides immediately (e.g. hexaconazole or carbendazim). Prune and burn infected leaves to prevent spore drift.";
                colorBorder = "var(--status-plfd)";
            } else if (leaf.final_diagnosis === "Other Disease") {
                cardClass = "warning";
                recIconText = "🍂";
                recTitleText = "Other Disease or Deficiency Detected";
                recBodyText = "Signs of non-PLFD leaf spots or nutrient deficiencies detected. Monitor leaf condition and apply general fungicide if symptoms spread.";
                colorBorder = "var(--status-other)";
            } else {
                cardClass = "pending";
                recIconText = "❓";
                recTitleText = "Stage 2 Diagnosis Pending";
                recBodyText = "Please complete Stage 2 training using your cropped datasets on Kaggle and place the weights inside the runs/ directory.";
                colorBorder = "var(--status-other)";
            }
            
            const cardHtml = `
                <div class="leaf-result-card ${cardClass}" style="border-left-color: ${colorBorder};">
                    <div class="leaf-card-header">
                        <span class="leaf-index-badge">Leaf #${leaf.index} (Detector: ${(leaf.box.confidence * 100).toFixed(0)}%)</span>
                        <span class="badge ${cardClass}">${diagnosisText} (${(leaf.confidence * 100).toFixed(1)}%)</span>
                    </div>
                    <div class="leaf-card-body">
                        <div class="leaf-thumbnail-container">
                            <img src="${leaf.crop_path ? leaf.crop_path : ''}" alt="Leaf #${leaf.index} Crop">
                        </div>
                        <div class="leaf-diagnosis-details">
                            <div class="progress-list">
                                <div class="progress-item">
                                    <div class="progress-labels">
                                        <span>Healthy Leaf</span>
                                        <span>${(healthyVal * 100).toFixed(1)}%</span>
                                    </div>
                                    <div class="progress-bar-bg">
                                        <div class="progress-bar-fill fill-healthy" style="width: ${healthyVal * 100}%"></div>
                                    </div>
                                </div>
                                <div class="progress-item">
                                    <div class="progress-labels">
                                        <span>Other Disease</span>
                                        <span>${(otherVal * 100).toFixed(1)}%</span>
                                    </div>
                                    <div class="progress-bar-bg">
                                        <div class="progress-bar-fill fill-other" style="width: ${otherVal * 100}%"></div>
                                    </div>
                                </div>
                                <div class="progress-item">
                                    <div class="progress-labels">
                                        <span>Pestalotiopsis Leaf Fall (PLFD)</span>
                                        <span>${(plfdVal * 100).toFixed(1)}%</span>
                                    </div>
                                    <div class="progress-bar-bg">
                                        <div class="progress-bar-fill fill-plfd" style="width: ${plfdVal * 100}%"></div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="leaf-recommendation">
                                <span class="rec-icon">${recIconText}</span>
                                <div class="rec-content">
                                    <h4 style="font-size: 0.8rem; font-weight: 600; margin: 0; color: #f8fafc;">${recTitleText}</h4>
                                    <p style="margin: 0.2rem 0 0 0; font-size: 0.75rem; color: var(--text-secondary); line-height: 1.3;">${recBodyText}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            multiResultsContainer.insertAdjacentHTML("beforeend", cardHtml);
        });
        
        lucide.createIcons();
    }

    function exportToCsv() {
        if (sessionHistory.length === 0) {
            alert("No leaf scans recorded in this session yet. Upload or capture some leaves first!");
            return;
        }
        
        let csvContent = "data:text/csv;charset=utf-8,";
        csvContent += "Timestamp,Latitude,Longitude,GPS Source,Google Maps URL,Filename,Leaf Index,Stage 1 Bounding Box Conf,Class ID,Diagnosis,Class Conf,Det Latency (ms),Cls Latency (ms),Total Latency (ms)\n";
        
        sessionHistory.forEach(row => {
            const r = [
                `"${row.timestamp}"`,
                row.latitude !== "" ? row.latitude : '""',
                row.longitude !== "" ? row.longitude : '""',
                `"${row.gps_source || ''}"`,
                `"${row.maps_url || ''}"`,
                `"${row.filename}"`,
                row.index,
                row.stage1_confidence.toFixed(4),
                row.class_id,
                `"${row.diagnosis}"`,
                row.diagnosis_confidence.toFixed(4),
                row.det_latency_ms,
                row.cls_latency_ms,
                row.total_latency
            ];
            csvContent += r.join(",") + "\n";
        });
        
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `rubber_leaf_analysis_${new Date().toISOString().slice(0,10)}.csv`);
        document.body.appendChild(link);
        
        link.click();
        document.body.removeChild(link);
    }

    if (exportCsvBtn) {
        exportCsvBtn.addEventListener("click", () => {
            exportToCsv();
        });
    }

    if (printReportBtn) {
        printReportBtn.addEventListener("click", () => {
            window.print();
        });
    }
});
