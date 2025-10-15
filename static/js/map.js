// NUTS-2 Map Visualization with D3.js
class MapVisualization {
    constructor(containerId, options = {}) {
        this.container = d3.select(`#${containerId}`);
        this.width = options.width || 800;
        this.height = options.height || 600;
        this.data = null;
        this.geoData = null;

        // Color scale
        this.colorScale = d3.scaleSequential(d3.interpolateBlues);

        // Region names translation (NUTS-2 and NUTS-0)
        this.regionNames = {
            // Lithuania
            'LT': 'Lietuva',
            'LT00': 'Lietuva',
            'LT01': 'Sostinės regionas',
            'LT02': 'Vidurio ir vakarų Lietuvos regionas',

            // Latvia
            'LV': 'Latvija',
            'LV00': 'Latvija',

            // Estonia
            'EE': 'Estija',
            'EE00': 'Estija',

            // Poland
            'PL': 'Lenkija',
            'PL00': 'Lenkija',

            // Albania
            'AL': 'Albanija',
            'AL01': 'Šiaurės Albanija',
            'AL02': 'Centrinė Albanija',
            'AL03': 'Pietų Albanija',

            // Other countries will use GeoJSON names
        };

        // Initialize tooltip
        this.tooltip = d3.select("body").append("div")
            .attr("class", "map-tooltip")
            .style("position", "absolute")
            .style("visibility", "hidden")
            .style("background-color", "white")
            .style("border", "1px solid #ddd")
            .style("border-radius", "5px")
            .style("padding", "10px")
            .style("box-shadow", "0 2px 4px rgba(0,0,0,0.2)")
            .style("pointer-events", "none")
            .style("z-index", "1000");

        this.init();
    }

    init() {
        // Create SVG
        this.svg = this.container.append("svg")
            .attr("width", this.width)
            .attr("height", this.height)
            .attr("class", "map-svg");

        // Create group for map elements
        this.g = this.svg.append("g");

        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([1, 8])
            .on("zoom", (event) => {
                this.g.attr("transform", event.transform);
            });

        this.svg.call(zoom);

        // Create projection (will be dynamically fitted in render)
        this.projection = d3.geoMercator();

        // Create path generator
        this.path = d3.geoPath().projection(this.projection);
    }

    async loadGeoJSON(url) {
        try {
            this.geoData = await d3.json(url);
            return this.geoData;
        } catch (error) {
            console.error("Error loading GeoJSON:", error);
            throw error;
        }
    }

    setData(data) {
        this.data = data;

        // Update color scale domain
        if (data && data.length > 0) {
            const values = data.map(d => d.value).filter(v => v !== null && v !== undefined);
            this.colorScale.domain([d3.min(values), d3.max(values)]);
        }
    }

    render() {
        if (!this.geoData) {
            console.error("No GeoJSON data loaded");
            return;
        }

        // Clear existing paths
        this.g.selectAll("path").remove();

        // Auto-fit projection to bounds
        this.fitProjectionToBounds();

        // Draw NUTS-2 regions
        const regions = this.g.selectAll("path")
            .data(this.geoData.features)
            .enter()
            .append("path")
            .attr("d", this.path)
            .attr("class", "region")
            .attr("fill", d => this.getRegionColor(d))
            .attr("stroke", "#333")
            .attr("stroke-width", 1)
            .style("cursor", "pointer")
            .on("mouseover", (event, d) => this.handleMouseOver(event, d))
            .on("mousemove", (event, d) => this.handleMouseMove(event, d))
            .on("mouseout", (event, d) => this.handleMouseOut(event, d))
            .on("click", (event, d) => this.handleClick(event, d));

        // Add hover effect
        regions
            .on("mouseenter", function() {
                d3.select(this)
                    .attr("stroke-width", 2)
                    .attr("stroke", "#ff6b6b")
                    .style("filter", "brightness(1.1)");
            })
            .on("mouseleave", function() {
                d3.select(this)
                    .attr("stroke-width", 1)
                    .attr("stroke", "#333")
                    .style("filter", "brightness(1)");
            });
    }

    fitProjectionToBounds() {
        if (!this.geoData || !this.geoData.features || this.geoData.features.length === 0) {
            return;
        }

        // Calculate bounds of all features
        const bounds = d3.geoBounds(this.geoData);

        // Calculate center
        const center = [
            (bounds[0][0] + bounds[1][0]) / 2,
            (bounds[0][1] + bounds[1][1]) / 2
        ];

        // Calculate scale using fitSize approach
        // This automatically scales to fit all features within the viewport
        const padding = 40; // pixels of padding
        const effectiveWidth = this.width - (padding * 2);
        const effectiveHeight = this.height - (padding * 2);

        // Use D3's fitSize to calculate optimal scale and translate
        this.projection.fitSize([effectiveWidth, effectiveHeight], this.geoData);

        // Adjust translate to account for padding
        const currentTranslate = this.projection.translate();
        this.projection.translate([
            currentTranslate[0] + padding,
            currentTranslate[1] + padding
        ]);

        // Update path generator
        this.path = d3.geoPath().projection(this.projection);

        console.log('Projection fitted to bounds:', {
            center: center,
            scale: this.projection.scale(),
            bounds: bounds,
            features: this.geoData.features.length
        });
    }

    getRegionColor(feature) {
        const regionId = feature.properties.NUTS_ID || feature.properties.id;

        if (!this.data) {
            return "#e0e0e0";
        }

        const regionData = this.data.find(d => d.region === regionId);

        if (!regionData) {
            return "#f5f5f5"; // Very light gray for regions not in dataset
        }

        // Check if region has value for selected indicator
        if (regionData.hasValue === false || regionData.value === null || regionData.value === undefined || isNaN(regionData.value)) {
            return "#ffcccc"; // Light red for missing indicator value
        }

        return this.colorScale(regionData.value);
    }

    handleMouseOver(event, feature) {
        this.tooltip.style("visibility", "visible");
        this.updateTooltipContent(feature);
    }

    handleMouseMove(event, feature) {
        this.tooltip
            .style("top", (event.pageY - 10) + "px")
            .style("left", (event.pageX + 10) + "px");
    }

    handleMouseOut(event, feature) {
        this.tooltip.style("visibility", "hidden");
    }

    handleClick(event, feature) {
        const regionId = feature.properties.NUTS_ID || feature.properties.id;
        console.log("Clicked region:", regionId, feature.properties);
    }

    updateTooltipContent(feature) {
        const regionId = feature.properties.NUTS_ID || feature.properties.id;

        // Try to get region name: first from our dictionary, then from GeoJSON properties
        let regionName = this.regionNames[regionId];
        if (!regionName) {
            regionName = feature.properties.NUTS_NAME || feature.properties.name || feature.properties.na || regionId;
        }

        let content = `<strong style="font-size: 15px;">${regionName}</strong><br>`;
        content += `<small style="color: #666;">NUTS-2 kodas: <strong>${regionId}</strong></small><br>`;

        if (this.data) {
            const regionData = this.data.find(d => d.region === regionId);

            if (regionData) {
                content += `<hr style="margin: 8px 0; border-color: #ddd;">`;

                // Show indicator value if available
                content += `<div style="margin: 5px 0;">`;
                content += `<strong style="color: #495057;">${regionData.label || 'Rodiklis'}:</strong><br>`;
                if (regionData.hasValue !== false && regionData.value !== null && regionData.value !== undefined && !isNaN(regionData.value)) {
                    content += `<span style="font-size: 16px; font-weight: bold; color: #007bff;">${regionData.value.toFixed(2)}</span>`;
                } else {
                    content += `<span style="color: #999; font-style: italic;">Nėra duomenų</span>`;
                }
                content += `</div>`;

                // Show missing percentage
                if (regionData.missingPercentage !== undefined) {
                    const missingColor = regionData.missingPercentage > 50 ? '#dc3545' :
                                        regionData.missingPercentage > 25 ? '#ffc107' : '#28a745';
                    content += `<hr style="margin: 8px 0; border-color: #ddd;">`;
                    content += `<div style="margin: 5px 0;">`;
                    content += `<strong style="color: #495057;">Trūkstamų reikšmių:</strong><br>`;
                    content += `<span style="color: ${missingColor}; font-weight: bold; font-size: 16px;">${regionData.missingPercentage}%</span>`;
                    content += `<br><small style="color: #666;">(${regionData.missingCount || 0} iš ${regionData.totalIndicators || 0} rodiklių)</small>`;
                    content += `</div>`;
                }
            } else {
                content += `<hr style="margin: 5px 0;">`;
                content += `<span style="color: #999;">Nėra duomenų</span>`;
            }
        }

        this.tooltip.html(content);
    }

    addLegend(options = {}) {
        const legendWidth = options.width || 250;
        const legendHeight = options.height || 20;
        const legendX = options.x || 20;
        const legendY = options.y || (this.height - 50);

        // Remove existing legend and defs
        this.svg.selectAll(".legend").remove();
        this.svg.selectAll("defs").remove();

        const legend = this.svg.append("g")
            .attr("class", "legend")
            .attr("transform", `translate(${legendX}, ${legendY})`);

        // Create gradient
        const defs = this.svg.append("defs");
        const linearGradient = defs.append("linearGradient")
            .attr("id", "legend-gradient")
            .attr("x1", "0%")
            .attr("y1", "0%")
            .attr("x2", "100%")
            .attr("y2", "0%");

        // Add color stops
        const numStops = 10;
        for (let i = 0; i <= numStops; i++) {
            const offset = (i / numStops) * 100;
            const value = this.colorScale.domain()[0] +
                         (this.colorScale.domain()[1] - this.colorScale.domain()[0]) * (i / numStops);
            linearGradient.append("stop")
                .attr("offset", `${offset}%`)
                .attr("stop-color", this.colorScale(value));
        }

        // Draw legend rectangle
        legend.append("rect")
            .attr("width", legendWidth)
            .attr("height", legendHeight)
            .style("fill", "url(#legend-gradient)")
            .style("stroke", "#333")
            .style("stroke-width", 1);

        // Add min/max labels
        legend.append("text")
            .attr("x", 0)
            .attr("y", legendHeight + 15)
            .style("font-size", "12px")
            .style("text-anchor", "start")
            .text(this.colorScale.domain()[0].toFixed(2));

        legend.append("text")
            .attr("x", legendWidth)
            .attr("y", legendHeight + 15)
            .style("font-size", "12px")
            .style("text-anchor", "end")
            .text(this.colorScale.domain()[1].toFixed(2));
    }

    destroy() {
        if (this.tooltip) {
            this.tooltip.remove();
        }
        if (this.svg) {
            this.svg.remove();
        }
    }
}

// Helper function to create map visualization
function createMapVisualization(containerId, geoJsonUrl, data, options = {}) {
    const map = new MapVisualization(containerId, options);

    return map.loadGeoJSON(geoJsonUrl)
        .then(() => {
            map.setData(data);
            map.render();

            if (options.showLegend !== false) {
                map.addLegend(options.legendOptions);
            }

            return map;
        })
        .catch(error => {
            console.error("Failed to create map visualization:", error);
            throw error;
        });
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { MapVisualization, createMapVisualization };
}
