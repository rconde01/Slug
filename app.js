// ============================================================
// Slug WebGPU Text Rendering Demo
// Based on the Slug algorithm by Eric Lengyel
// ============================================================

// ============================================================
// Section 1: Math Utilities (column-major mat4)
// ============================================================

function mat4Identity() {
    return new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]);
}

function mat4Multiply(a, b) {
    const r = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
        for (let row = 0; row < 4; row++) {
            let sum = 0;
            for (let k = 0; k < 4; k++) {
                sum += a[k * 4 + row] * b[c * 4 + k];
            }
            r[c * 4 + row] = sum;
        }
    }
    return r;
}

function mat4Translate(tx, ty, tz) {
    return new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        tx, ty, tz, 1
    ]);
}

function mat4Scale(sx, sy, sz) {
    return new Float32Array([
        sx, 0, 0, 0,
        0, sy, 0, 0,
        0, 0, sz, 0,
        0, 0, 0, 1
    ]);
}

function mat4Perspective(fovY, aspect, near, far) {
    const f = 1.0 / Math.tan(fovY / 2);
    const nf = near - far;
    return new Float32Array([
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, far / nf, -1,
        0, 0, near * far / nf, 0
    ]);
}

function quatIdentity() { return [0, 0, 0, 1]; }

function quatMultiply(a, b) {
    return [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]
    ];
}

function quatNormalize(q) {
    const len = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (len < 1e-10) return [0, 0, 0, 1];
    return [q[0] / len, q[1] / len, q[2] / len, q[3] / len];
}

function quatToMat4(q) {
    const [x, y, z, w] = q;
    const x2 = x + x, y2 = y + y, z2 = z + z;
    const xx = x * x2, xy = x * y2, xz = x * z2;
    const yy = y * y2, yz = y * z2, zz = z * z2;
    const wx = w * x2, wy = w * y2, wz = w * z2;
    return new Float32Array([
        1 - (yy + zz), xy + wz, xz - wy, 0,
        xy - wz, 1 - (xx + zz), yz + wx, 0,
        xz + wy, yz - wx, 1 - (xx + yy), 0,
        0, 0, 0, 1
    ]);
}

function vec3Cross(a, b) {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ];
}

function vec3Dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function vec3Length(v) {
    return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

// ============================================================
// Section 2: Trackball
// ============================================================

class Trackball {
    constructor(canvas) {
        this.canvas = canvas;
        this.rotation = quatIdentity();
        this.dragging = false;
        this.lastPoint = null;
        this.zoom = 5.0;

        canvas.addEventListener('pointerdown', (e) => this.onDown(e));
        canvas.addEventListener('pointermove', (e) => this.onMove(e));
        canvas.addEventListener('pointerup', () => this.onUp());
        canvas.addEventListener('pointerleave', () => this.onUp());
        canvas.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });
    }

    projectToSphere(clientX, clientY) {
        const rect = this.canvas.getBoundingClientRect();
        const r = Math.min(rect.width, rect.height) / 2;
        const px = (clientX - rect.left - rect.width / 2) / r;
        const py = -(clientY - rect.top - rect.height / 2) / r;
        const d = px * px + py * py;
        if (d <= 1) return [px, py, Math.sqrt(1 - d)];
        const s = 1 / Math.sqrt(d);
        return [px * s, py * s, 0];
    }

    onDown(e) {
        this.dragging = true;
        this.lastPoint = this.projectToSphere(e.clientX, e.clientY);
        this.canvas.setPointerCapture(e.pointerId);
    }

    onMove(e) {
        if (!this.dragging) return;
        const cur = this.projectToSphere(e.clientX, e.clientY);
        const prev = this.lastPoint;
        const axis = vec3Cross(prev, cur);
        const len = vec3Length(axis);
        if (len < 1e-8) return;
        const dot = vec3Dot(prev, cur);
        const q = quatNormalize([axis[0], axis[1], axis[2], 1 + dot]);
        this.rotation = quatNormalize(quatMultiply(q, this.rotation));
        this.lastPoint = cur;
    }

    onUp() {
        this.dragging = false;
    }

    onWheel(e) {
        e.preventDefault();
        this.zoom *= e.deltaY > 0 ? 1.1 : 0.9;
        this.zoom = Math.max(0.01, Math.min(1000, this.zoom));
    }

    getRotationMatrix() {
        return quatToMat4(this.rotation);
    }
}

// ============================================================
// Section 3: Font Processing
// ============================================================

function extractQuadraticCurves(glyph) {
    if (!glyph.path || !glyph.path.commands) return [];
    const commands = glyph.path.commands;
    if (commands.length === 0) return [];

    // Log command types for debugging
    const cmdTypes = commands.map(c => c.type).join('');
    const glyphName = glyph.name || `#${glyph.index}`;
    console.log(`Glyph '${glyphName}' commands: ${cmdTypes}`);

    const curves = [];
    let cx = 0, cy = 0;
    let startX = 0, startY = 0;

    for (const cmd of commands) {
        switch (cmd.type) {
            case 'M':
                startX = cmd.x;
                startY = cmd.y;
                cx = cmd.x;
                cy = cmd.y;
                break;
            case 'L': {
                // Convert line segment to degenerate quadratic with p2 = p1.
                // Using p1 (not midpoint) prevents calcRootCode sign-bit instability:
                // with midpoint, p2.y sits ON the line, so its sign relative to the
                // pixel flips at the line's midpoint, causing noise between adjacent pixels.
                curves.push({ p1: { x: cx, y: cy }, p2: { x: cx, y: cy }, p3: { x: cmd.x, y: cmd.y } });
                cx = cmd.x;
                cy = cmd.y;
                break;
            }
            case 'Q':
                curves.push({ p1: { x: cx, y: cy }, p2: { x: cmd.x1, y: cmd.y1 }, p3: { x: cmd.x, y: cmd.y } });
                cx = cmd.x;
                cy = cmd.y;
                break;
            case 'C': {
                // Approximate cubic with 2 quadratics by splitting at t=0.5.
                const quads = cubicToQuadratics(
                    { x: cx, y: cy },
                    { x: cmd.x1, y: cmd.y1 },
                    { x: cmd.x2, y: cmd.y2 },
                    { x: cmd.x, y: cmd.y }
                );
                curves.push(...quads);
                cx = cmd.x;
                cy = cmd.y;
                break;
            }
            case 'Z':
                if (Math.abs(cx - startX) > 0.01 || Math.abs(cy - startY) > 0.01) {
                    // Use p2 = p1 for closing segments too (same stability fix as L).
                    curves.push({ p1: { x: cx, y: cy }, p2: { x: cx, y: cy }, p3: { x: startX, y: startY } });
                }
                cx = startX;
                cy = startY;
                break;
        }
    }
    return curves;
}

// Split a cubic Bezier at t=0.5 and approximate each half as a quadratic.
function cubicToQuadratics(p0, c0, c1, p1) {
    // De Casteljau split at t=0.5
    const m01 = mid(p0, c0);
    const m12 = mid(c0, c1);
    const m23 = mid(c1, p1);
    const m012 = mid(m01, m12);
    const m123 = mid(m12, m23);
    const m0123 = mid(m012, m123);

    // Approximate left half (p0, m01, m012, m0123) as quadratic
    const qL = quadFromCubic(p0, m01, m012, m0123);
    // Approximate right half (m0123, m123, m23, p1) as quadratic
    const qR = quadFromCubic(m0123, m123, m23, p1);
    return [qL, qR];
}

function mid(a, b) {
    return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

// Best-fit quadratic for cubic (a, b, c, d): control point = (3(b+c) - a - d) / 4
function quadFromCubic(a, b, c, d) {
    return {
        p1: a,
        p2: { x: (3 * (b.x + c.x) - a.x - d.x) / 4, y: (3 * (b.y + c.y) - a.y - d.y) / 4 },
        p3: d
    };
}

// ============================================================
// Section 4: Slug Data Builder
// ============================================================

const BAND_TEX_WIDTH = 4096;

class SlugDataBuilder {
    constructor() {
        this.curveTexData = []; // rows of Float32Array data
        this.bandTexData = [];  // rows of Uint32Array data
        this.curveTexRow = 0;
        this.curveTexX = 0;
        this.bandTexRow = 0;
        this.glyphCache = new Map();
    }

    processGlyph(glyph, unitsPerEm) {
        const key = glyph.index;
        if (this.glyphCache.has(key)) return this.glyphCache.get(key);

        const curves = extractQuadraticCurves(glyph);
        if (curves.length === 0) {
            const meta = {
                curves: [],
                glyphLocX: 0, glyphLocY: 0,
                bandMaxX: 0, bandMaxY: 0,
                bandTransform: [1, 1, 0, 0],
                bbox: null,
                advanceWidth: (glyph.advanceWidth || 0) / unitsPerEm
            };
            this.glyphCache.set(key, meta);
            return meta;
        }

        // Compute bounding box in raw font units
        let xMin = Infinity, yMin = Infinity, xMax = -Infinity, yMax = -Infinity;
        for (const c of curves) {
            for (const p of [c.p1, c.p2, c.p3]) {
                xMin = Math.min(xMin, p.x);
                yMin = Math.min(yMin, p.y);
                xMax = Math.max(xMax, p.x);
                yMax = Math.max(yMax, p.y);
            }
        }

        // Add padding for anti-aliasing (in font units).
        // Use 5% of glyph size or minimum 20 font units, whichever is larger.
        const pad = Math.max(Math.max(xMax - xMin, yMax - yMin) * 0.05, 20);
        const bboxPadded = { xMin: xMin - pad, yMin: yMin - pad, xMax: xMax + pad, yMax: yMax + pad };

        // Choose band counts based on curve complexity
        const numHBands = Math.max(2, Math.min(16, Math.ceil(Math.sqrt(curves.length))));
        const numVBands = Math.max(2, Math.min(16, Math.ceil(Math.sqrt(curves.length))));

        // Write curves to curve texture
        const curveLocs = [];
        this._ensureCurveRow(this.curveTexRow);

        // Check if we need a new row
        if (this.curveTexX + curves.length * 2 > BAND_TEX_WIDTH) {
            this.curveTexRow++;
            this.curveTexX = 0;
        }
        this._ensureCurveRow(this.curveTexRow);

        for (const c of curves) {
            const locX = this.curveTexX;
            const locY = this.curveTexRow;
            curveLocs.push({ x: locX, y: locY });

            const row = this.curveTexData[locY];
            const off = locX * 4;
            row[off] = c.p1.x; row[off + 1] = c.p1.y; row[off + 2] = c.p2.x; row[off + 3] = c.p2.y;
            const off2 = (locX + 1) * 4;
            row[off2] = c.p3.x; row[off2 + 1] = c.p3.y; row[off2 + 2] = 0; row[off2 + 3] = 0;

            this.curveTexX += 2;
        }

        // Assign curves to bands
        const bandHeight = (yMax - yMin) / numHBands;
        const bandWidth = (xMax - xMin) / numVBands;

        const hBands = Array.from({ length: numHBands }, () => []);
        const vBands = Array.from({ length: numVBands }, () => []);

        for (let i = 0; i < curves.length; i++) {
            const c = curves[i];
            const cyMin = Math.min(c.p1.y, c.p2.y, c.p3.y);
            const cyMax = Math.max(c.p1.y, c.p2.y, c.p3.y);
            const cxMin = Math.min(c.p1.x, c.p2.x, c.p3.x);
            const cxMax = Math.max(c.p1.x, c.p2.x, c.p3.x);

            // Horizontal bands (indexed by y)
            const hStart = Math.max(0, Math.floor((cyMin - yMin) / bandHeight));
            const hEnd = Math.min(numHBands - 1, Math.floor((cyMax - yMin) / bandHeight));
            for (let b = hStart; b <= hEnd; b++) hBands[b].push(i);

            // Vertical bands (indexed by x)
            const vStart = Math.max(0, Math.floor((cxMin - xMin) / bandWidth));
            const vEnd = Math.min(numVBands - 1, Math.floor((cxMax - xMin) / bandWidth));
            for (let b = vStart; b <= vEnd; b++) vBands[b].push(i);
        }

        // Sort: horizontal bands by descending max x, vertical by descending max y
        for (const band of hBands) {
            band.sort((a, b) => {
                const maxA = Math.max(curves[a].p1.x, curves[a].p2.x, curves[a].p3.x);
                const maxB = Math.max(curves[b].p1.x, curves[b].p2.x, curves[b].p3.x);
                return maxB - maxA;
            });
        }
        for (const band of vBands) {
            band.sort((a, b) => {
                const maxA = Math.max(curves[a].p1.y, curves[a].p2.y, curves[a].p3.y);
                const maxB = Math.max(curves[b].p1.y, curves[b].p2.y, curves[b].p3.y);
                return maxB - maxA;
            });
        }

        // Pack band texture data (one row per glyph)
        const glyphLocX = 0;
        const glyphLocY = this.bandTexRow;
        this._ensureBandRow(glyphLocY);
        const bandRow = this.bandTexData[glyphLocY];

        const numHeaders = numHBands + numVBands;
        let listOffset = numHeaders;

        // Write horizontal band headers
        for (let b = 0; b < numHBands; b++) {
            const off = (glyphLocX + b) * 4;
            bandRow[off] = hBands[b].length;
            bandRow[off + 1] = listOffset;
            bandRow[off + 2] = 0;
            bandRow[off + 3] = 0;

            // Write curve list
            for (let j = 0; j < hBands[b].length; j++) {
                const ci = hBands[b][j];
                const loc = curveLocs[ci];
                const loff = (glyphLocX + listOffset + j) * 4;
                bandRow[loff] = loc.x;
                bandRow[loff + 1] = loc.y;
                bandRow[loff + 2] = 0;
                bandRow[loff + 3] = 0;
            }
            listOffset += hBands[b].length;
        }

        // Write vertical band headers
        for (let b = 0; b < numVBands; b++) {
            const off = (glyphLocX + numHBands + b) * 4;
            bandRow[off] = vBands[b].length;
            bandRow[off + 1] = listOffset;
            bandRow[off + 2] = 0;
            bandRow[off + 3] = 0;

            for (let j = 0; j < vBands[b].length; j++) {
                const ci = vBands[b][j];
                const loc = curveLocs[ci];
                const loff = (glyphLocX + listOffset + j) * 4;
                bandRow[loff] = loc.x;
                bandRow[loff + 1] = loc.y;
                bandRow[loff + 2] = 0;
                bandRow[loff + 3] = 0;
            }
            listOffset += vBands[b].length;
        }

        this.bandTexRow++;

        // Compute band transform (guard against zero-size bbox)
        const dx = Math.max(xMax - xMin, 0.001);
        const dy = Math.max(yMax - yMin, 0.001);
        const bandScaleX = numVBands / dx;
        const bandScaleY = numHBands / dy;
        const bandOffsetX = -xMin * bandScaleX;
        const bandOffsetY = -yMin * bandScaleY;

        // Verify band data integrity
        const totalListEntries = listOffset - numHeaders;
        for (let b = 0; b < numHBands; b++) {
            const count = bandRow[(glyphLocX + b) * 4];
            const off = bandRow[(glyphLocX + b) * 4 + 1];
            for (let j = 0; j < count; j++) {
                const cx = bandRow[(glyphLocX + off + j) * 4];
                const cy = bandRow[(glyphLocX + off + j) * 4 + 1];
                if (cx >= this.curveTexX + 2 || cy > this.curveTexRow) {
                    console.error(`INVALID curve loc in hband ${b}: (${cx},${cy}), curveTexX=${this.curveTexX}, curveTexRow=${this.curveTexRow}`);
                }
            }
        }

        const glyphName = glyph.name || `glyph#${glyph.index}`;
        console.log(`Glyph '${glyphName}': ${curves.length} curves, ${numHBands}h x ${numVBands}v bands, bbox=[${xMin.toFixed(0)},${yMin.toFixed(0)},${xMax.toFixed(0)},${yMax.toFixed(0)}], bandRow=${glyphLocY}`);

        const meta = {
            curves,
            glyphLocX, glyphLocY,
            bandMaxX: numVBands - 1,
            bandMaxY: numHBands - 1,
            bandTransform: [bandScaleX, bandScaleY, bandOffsetX, bandOffsetY],
            bbox: bboxPadded,
            advanceWidth: (glyph.advanceWidth || 0) / unitsPerEm
        };
        this.glyphCache.set(key, meta);
        return meta;
    }

    _ensureCurveRow(y) {
        while (this.curveTexData.length <= y) {
            this.curveTexData.push(new Float32Array(BAND_TEX_WIDTH * 4));
        }
    }

    _ensureBandRow(y) {
        while (this.bandTexData.length <= y) {
            this.bandTexData.push(new Uint32Array(BAND_TEX_WIDTH * 4));
        }
    }

    getCurveTexHeight() {
        return Math.max(1, this.curveTexData.length);
    }

    getBandTexHeight() {
        return Math.max(1, this.bandTexData.length);
    }

    getCurveTexArray() {
        const h = this.getCurveTexHeight();
        const data = new Float32Array(BAND_TEX_WIDTH * 4 * h);
        for (let y = 0; y < this.curveTexData.length; y++) {
            data.set(this.curveTexData[y], y * BAND_TEX_WIDTH * 4);
        }
        return data;
    }

    getBandTexArray() {
        const h = this.getBandTexHeight();
        const data = new Uint32Array(BAND_TEX_WIDTH * 4 * h);
        for (let y = 0; y < this.bandTexData.length; y++) {
            data.set(this.bandTexData[y], y * BAND_TEX_WIDTH * 4);
        }
        return data;
    }
}

// ============================================================
// Section 5: Text Layout & Vertex Generation
// ============================================================

const VERTEX_STRIDE = 64; // bytes per vertex

function layoutAndBuildVertices(text, font, slugBuilder) {
    const upm = font.unitsPerEm;
    const scale = 1.0 / upm;
    const glyphs = font.stringToGlyphs(text);

    // First pass: process all glyphs and compute layout
    const quads = [];
    let cursorX = 0;

    for (let i = 0; i < glyphs.length; i++) {
        const glyph = glyphs[i];
        const meta = slugBuilder.processGlyph(glyph, upm);

        if (meta.curves.length > 0 && meta.bbox) {
            quads.push({ meta, x: cursorX });
        }
        cursorX += meta.advanceWidth;
    }

    // Center text horizontally and vertically
    const totalWidth = cursorX;
    const offsetX = -totalWidth / 2;
    // Center vertically using font ascender/descender
    const ascender = (font.ascender || 800) / upm;
    const descender = (font.descender || -200) / upm;
    const offsetY = -(ascender + descender) / 2;

    if (quads.length === 0) return { vertexData: null, indexData: null, indexCount: 0 };

    // Build vertex and index buffers
    const vertexCount = quads.length * 4;
    const indexCount = quads.length * 6;
    const vertexBuf = new ArrayBuffer(VERTEX_STRIDE * vertexCount);
    const vView = new DataView(vertexBuf);
    const indexData = new Uint16Array(indexCount);

    for (let qi = 0; qi < quads.length; qi++) {
        const { meta, x } = quads[qi];
        const bbox = meta.bbox;
        const worldX = x + offsetX;

        // Object-space quad corners (world coordinates)
        const x0 = worldX + bbox.xMin * scale;
        const y0 = offsetY + bbox.yMin * scale;
        const x1 = worldX + bbox.xMax * scale;
        const y1 = offsetY + bbox.yMax * scale;

        // Em-space texcoords (raw font units with padding)
        const u0 = bbox.xMin;
        const v0 = bbox.yMin;
        const u1 = bbox.xMax;
        const v1 = bbox.yMax;

        const corners = [
            { px: x0, py: y0, tu: u0, tv: v0 }, // bottom-left
            { px: x1, py: y0, tu: u1, tv: v0 }, // bottom-right
            { px: x1, py: y1, tu: u1, tv: v1 }, // top-right
            { px: x0, py: y1, tu: u0, tv: v1 }, // top-left
        ];

        const baseVertex = qi * 4;
        for (let vi = 0; vi < 4; vi++) {
            const off = (baseVertex + vi) * VERTEX_STRIDE;
            const c = corners[vi];
            // position (float32x2)
            vView.setFloat32(off + 0, c.px, true);
            vView.setFloat32(off + 4, c.py, true);
            // texcoord (float32x2)
            vView.setFloat32(off + 8, c.tu, true);
            vView.setFloat32(off + 12, c.tv, true);
            // glyph_xy (sint32x2)
            vView.setInt32(off + 16, meta.glyphLocX, true);
            vView.setInt32(off + 20, meta.glyphLocY, true);
            // band_max_flags (sint32x2)
            vView.setInt32(off + 24, meta.bandMaxX, true);
            vView.setInt32(off + 28, meta.bandMaxY, true);
            // band_transform (float32x4)
            vView.setFloat32(off + 32, meta.bandTransform[0], true);
            vView.setFloat32(off + 36, meta.bandTransform[1], true);
            vView.setFloat32(off + 40, meta.bandTransform[2], true);
            vView.setFloat32(off + 44, meta.bandTransform[3], true);
            // color (float32x4) - white by default
            vView.setFloat32(off + 48, 1.0, true);
            vView.setFloat32(off + 52, 1.0, true);
            vView.setFloat32(off + 56, 1.0, true);
            vView.setFloat32(off + 60, 1.0, true);
        }

        // Two triangles per quad
        const idx = qi * 6;
        indexData[idx + 0] = baseVertex + 0;
        indexData[idx + 1] = baseVertex + 1;
        indexData[idx + 2] = baseVertex + 2;
        indexData[idx + 3] = baseVertex + 0;
        indexData[idx + 4] = baseVertex + 2;
        indexData[idx + 5] = baseVertex + 3;
    }

    return { vertexData: vertexBuf, indexData, indexCount };
}

// ============================================================
// Section 6: WebGPU Renderer
// ============================================================

class SlugRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.device = null;
        this.context = null;
        this.pipeline = null;
        this.bindGroupLayout = null;
        this.uniformBuffer = null;
        this.vertexBuffer = null;
        this.indexBuffer = null;
        this.curveTexture = null;
        this.bandTexture = null;
        this.bindGroup = null;
        this.indexCount = 0;
        this.canvasFormat = null;
        this.textColor = [1, 1, 1, 1];
    }

    async init() {
        if (!navigator.gpu) throw new Error('WebGPU not supported in this browser.');

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error('No WebGPU adapter found.');

        this.device = await adapter.requestDevice();
        this.context = this.canvas.getContext('webgpu');
        this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: this.canvasFormat,
            alphaMode: 'premultiplied',
        });

        // Load shader (cache-bust to avoid stale cached versions)
        const shaderCode = await (await fetch('shaders.wgsl?v=' + Date.now())).text();
        const shaderModule = this.device.createShaderModule({ code: shaderCode });

        // Check for compilation errors
        const info = await shaderModule.getCompilationInfo();
        for (const msg of info.messages) {
            if (msg.type === 'error') {
                throw new Error(`Shader error: ${msg.message} (line ${msg.lineNum})`);
            }
        }

        // Bind group layout
        this.bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'uint', viewDimension: '2d' } },
            ]
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [this.bindGroupLayout]
        });

        // Vertex buffer layout
        const vertexBufferLayout = {
            arrayStride: VERTEX_STRIDE,
            attributes: [
                { shaderLocation: 0, offset: 0, format: 'float32x2' },   // position
                { shaderLocation: 1, offset: 8, format: 'float32x2' },   // texcoord
                { shaderLocation: 2, offset: 16, format: 'sint32x2' },   // glyph_xy
                { shaderLocation: 3, offset: 24, format: 'sint32x2' },   // band_max_flags
                { shaderLocation: 4, offset: 32, format: 'float32x4' },  // band_transform
                { shaderLocation: 5, offset: 48, format: 'float32x4' },  // color
            ]
        };

        // Render pipeline
        this.pipeline = this.device.createRenderPipeline({
            layout: pipelineLayout,
            vertex: {
                module: shaderModule,
                entryPoint: 'vs_main',
                buffers: [vertexBufferLayout]
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: this.canvasFormat,
                    blend: {
                        color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
                        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' }
                    }
                }]
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'none'
            },
        });

        // Uniform buffer (mat4x4 + vec4 = 80 bytes)
        this.uniformBuffer = this.device.createBuffer({
            size: 80,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Create placeholder 1x1 textures
        this._createPlaceholderTextures();
    }

    _createPlaceholderTextures() {
        this.curveTexture = this.device.createTexture({
            size: [1, 1],
            format: 'rgba32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.bandTexture = this.device.createTexture({
            size: [1, 1],
            format: 'rgba32uint',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this._updateBindGroup();
    }

    _updateBindGroup() {
        this.bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: this.curveTexture.createView() },
                { binding: 2, resource: this.bandTexture.createView() },
            ]
        });
    }

    uploadSlugData(slugBuilder) {
        const curveH = slugBuilder.getCurveTexHeight();
        const bandH = slugBuilder.getBandTexHeight();

        // Destroy old textures
        if (this.curveTexture) this.curveTexture.destroy();
        if (this.bandTexture) this.bandTexture.destroy();

        // Create curve texture
        this.curveTexture = this.device.createTexture({
            size: [BAND_TEX_WIDTH, curveH],
            format: 'rgba32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        const curveData = slugBuilder.getCurveTexArray();
        this.device.queue.writeTexture(
            { texture: this.curveTexture },
            curveData,
            { bytesPerRow: BAND_TEX_WIDTH * 16 },
            { width: BAND_TEX_WIDTH, height: curveH }
        );

        // Create band texture
        this.bandTexture = this.device.createTexture({
            size: [BAND_TEX_WIDTH, bandH],
            format: 'rgba32uint',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        const bandData = slugBuilder.getBandTexArray();
        this.device.queue.writeTexture(
            { texture: this.bandTexture },
            bandData,
            { bytesPerRow: BAND_TEX_WIDTH * 16 },
            { width: BAND_TEX_WIDTH, height: bandH }
        );

        this._updateBindGroup();
    }

    uploadVertices(vertexData, indexData, indexCount) {
        if (this.vertexBuffer) this.vertexBuffer.destroy();
        if (this.indexBuffer) this.indexBuffer.destroy();

        if (!vertexData || indexCount === 0) {
            this.vertexBuffer = null;
            this.indexBuffer = null;
            this.indexCount = 0;
            return;
        }

        this.vertexBuffer = this.device.createBuffer({
            size: vertexData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.vertexBuffer, 0, vertexData);

        this.indexBuffer = this.device.createBuffer({
            size: indexData.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.indexBuffer, 0, indexData);

        this.indexCount = indexCount;
    }

    render(trackball) {
        // Resize canvas to match display size
        const dpr = window.devicePixelRatio || 1;
        const displayW = Math.floor(this.canvas.clientWidth * dpr);
        const displayH = Math.floor(this.canvas.clientHeight * dpr);
        if (displayW === 0 || displayH === 0) return;
        if (this.canvas.width !== displayW || this.canvas.height !== displayH) {
            this.canvas.width = displayW;
            this.canvas.height = displayH;
        }

        const aspect = displayW / displayH;
        const fovY = Math.PI / 4;

        // MVP = projection * view * model
        const projection = mat4Perspective(fovY, aspect, 0.01, 2000);
        const view = mat4Translate(0, 0, -trackball.zoom);
        const model = trackball.getRotationMatrix();
        const mvp = mat4Multiply(projection, mat4Multiply(view, model));

        // Upload uniforms
        const uniformData = new Float32Array(20);
        uniformData.set(mvp, 0);
        uniformData.set([displayW, displayH, this.debugMode || 0, 0], 16);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        // Render
        const commandEncoder = this.device.createCommandEncoder();
        const passDesc = {
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0.1, g: 0.1, b: 0.12, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }]
        };

        const pass = commandEncoder.beginRenderPass(passDesc);
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);

        if (this.vertexBuffer && this.indexBuffer && this.indexCount > 0) {
            pass.setVertexBuffer(0, this.vertexBuffer);
            pass.setIndexBuffer(this.indexBuffer, 'uint16');
            pass.drawIndexed(this.indexCount);
        }

        pass.end();
        this.device.queue.submit([commandEncoder.finish()]);
    }
}

// ============================================================
// Section 7: Main Application
// ============================================================

async function main() {
    const canvas = document.getElementById('canvas');
    const textInput = document.getElementById('textInput');
    const fontInput = document.getElementById('fontInput');
    const colorPicker = document.getElementById('colorPicker');
    const overlay = document.getElementById('overlay');
    const statusEl = document.getElementById('status');

    function showError(msg) {
        overlay.textContent = msg;
        overlay.className = 'error';
    }

    function showInfo(msg) {
        overlay.textContent = msg;
        overlay.className = 'info';
    }

    function hideOverlay() {
        overlay.className = 'hidden';
    }

    function setStatus(msg) {
        statusEl.textContent = msg;
    }

    // Check opentype.js is loaded
    if (typeof opentype === 'undefined') {
        showError('opentype.js failed to load. Check your internet connection and reload.');
        return;
    }

    // Initialize renderer
    let renderer;
    try {
        renderer = new SlugRenderer(canvas);
        await renderer.init();
        showInfo('Loading font...');
    } catch (e) {
        showError(`Initialization failed: ${e.message}`);
        console.error(e);
        return;
    }

    const trackball = new Trackball(canvas);
    let currentFont = null;

    // Update text rendering
    function updateText() {
        if (!currentFont) return;

        const text = textInput.value || '';
        if (text.trim().length === 0) {
            renderer.uploadVertices(null, null, 0);
            setStatus('');
            return;
        }

        try {
            const slugBuilder = new SlugDataBuilder();
            const { vertexData, indexData, indexCount } = layoutAndBuildVertices(text, currentFont, slugBuilder);

            if (indexCount > 0) {
                // Apply text color to vertex data
                const hex = colorPicker.value;
                const r = parseInt(hex.slice(1, 3), 16) / 255;
                const g = parseInt(hex.slice(3, 5), 16) / 255;
                const b = parseInt(hex.slice(5, 7), 16) / 255;
                const view = new DataView(vertexData);
                for (let i = 0; i < indexCount / 6 * 4; i++) {
                    const off = i * VERTEX_STRIDE;
                    view.setFloat32(off + 48, r, true);
                    view.setFloat32(off + 52, g, true);
                    view.setFloat32(off + 56, b, true);
                    view.setFloat32(off + 60, 1.0, true);
                }
                renderer.uploadSlugData(slugBuilder);
                renderer.uploadVertices(vertexData, indexData, indexCount);
            } else {
                renderer.uploadVertices(null, null, 0);
            }

            const numGlyphs = indexCount / 6;
            const numCurves = slugBuilder.curveTexX / 2;
            setStatus(`${numGlyphs} glyphs, ${numCurves} curves, tex: ${slugBuilder.getCurveTexHeight()}x${slugBuilder.getBandTexHeight()} rows`);
        } catch (e) {
            console.error('Text update error:', e);
            setStatus(`Error: ${e.message}`);
        }
    }

    // Load font from ArrayBuffer
    function loadFontFromBuffer(buffer, name) {
        try {
            currentFont = opentype.parse(buffer);
            hideOverlay();
            setStatus(`Font: ${currentFont.names.fontFamily?.en || name} (${currentFont.numGlyphs} glyphs, UPM=${currentFont.unitsPerEm})`);
            updateText();
        } catch (e) {
            showError(`Failed to parse font: ${e.message}`);
            console.error(e);
        }
    }

    // Try loading a default font from CDN
    async function loadDefaultFont() {
        const urls = [
            'https://cdn.jsdelivr.net/npm/@fontsource/roboto/files/roboto-latin-400-normal.woff',
            'https://cdn.jsdelivr.net/npm/@fontsource/inter/files/inter-latin-400-normal.woff',
            'https://cdn.jsdelivr.net/npm/@fontsource/noto-sans/files/noto-sans-latin-400-normal.woff',
        ];

        for (const url of urls) {
            try {
                const resp = await fetch(url);
                if (!resp.ok) continue;
                const buffer = await resp.arrayBuffer();
                loadFontFromBuffer(buffer, url.split('/').pop());
                return true;
            } catch (e) {
                console.warn(`Failed to load font from ${url}:`, e);
            }
        }
        return false;
    }

    // Try to load default font
    const loaded = await loadDefaultFont();
    if (!loaded) {
        showInfo('Please select a .ttf font file using the "Font" button above.');
    }

    // Debug mode toggle: press D to cycle through debug views
    const debugModes = ['Normal', 'Solid quads', 'H-coverage only', 'V-coverage only'];
    document.addEventListener('keydown', (e) => {
        if (e.key === 'd' || e.key === 'D') {
            renderer.debugMode = ((renderer.debugMode || 0) + 1) % debugModes.length;
            setStatus(`Debug: ${debugModes[renderer.debugMode]}`);
        }
    });

    // Event handlers
    textInput.addEventListener('input', () => updateText());
    colorPicker.addEventListener('input', () => updateText());

    fontInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = () => loadFontFromBuffer(reader.result, file.name);
        reader.readAsArrayBuffer(file);
    });

    // Render loop
    function frame() {
        renderer.render(trackball);
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

main().catch(e => {
    console.error('Fatal error:', e);
    const overlay = document.getElementById('overlay');
    if (overlay) {
        overlay.textContent = `Fatal error: ${e.message}`;
        overlay.className = 'error';
    }
});
