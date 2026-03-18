// ============================================================
// Slug GPU Text Rendering - WebGPU WGSL Port
// Based on the Slug algorithm by Eric Lengyel
// Original HLSL reference: MIT License, Copyright 2017
// ============================================================

// Band texture uses a fixed width of 4096 texels.
const BAND_TEX_LOG_W: u32 = 12u;

// --- Bindings ---

struct Uniforms {
    mvp: mat4x4<f32>,
    viewport: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var curveTexture: texture_2d<f32>;
@group(0) @binding(2) var bandTexture: texture_2d<u32>;

// --- Inter-stage structures ---

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) texcoord: vec2<f32>,
    @location(2) @interpolate(flat) banding: vec4<f32>,
    @location(3) @interpolate(flat) glyph: vec4<i32>,
};

// --- Vertex Shader (simplified, no dynamic dilation) ---

@vertex
fn vs_main(
    @location(0) position: vec2<f32>,
    @location(1) texcoord: vec2<f32>,
    @location(2) glyph_xy: vec2<i32>,
    @location(3) band_max_flags: vec2<i32>,
    @location(4) band_transform: vec4<f32>,
    @location(5) color: vec4<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.mvp * vec4<f32>(position, 0.0, 1.0);
    out.color = color;
    out.texcoord = texcoord;
    out.banding = band_transform;
    out.glyph = vec4<i32>(glyph_xy, band_max_flags);
    return out;
}

// --- Fragment Shader Helper Functions ---

// Calculate root eligibility code for a quadratic Bezier curve.
// Uses sign bits of the three control point y-coordinates to index
// a lookup table that determines which roots contribute to coverage.
fn calcRootCode(y1: f32, y2: f32, y3: f32) -> u32 {
    // Extract sign bits of the three control point coordinates.
    // i1 bit 0 = sign of y1, i2 bit 1 = sign of y2, i3 bit 2 = sign of y3.
    let i1 = bitcast<u32>(y1) >> 31u;
    let i2 = bitcast<u32>(y2) >> 30u;
    let i3 = bitcast<u32>(y3) >> 29u;

    // Pack sign bits into a 3-bit index.
    let shift = (i3 & 4u) | (i2 & 2u) | (i1 & 1u);

    // Eligibility returned in bits 0 and 8.
    return (0x2E74u >> shift) & 0x0101u;
}

// Solve for x-coordinates where the curve crosses y = 0.
// The quadratic polynomial is: a*t^2 - 2*b*t + c
fn solveHorizPoly(p12: vec4<f32>, p3: vec2<f32>) -> vec2<f32> {
    let a = vec2<f32>(p12.x - p12.z * 2.0 + p3.x, p12.y - p12.w * 2.0 + p3.y);
    let b = vec2<f32>(p12.x - p12.z, p12.y - p12.w);
    let ra = 1.0 / a.y;
    let rb = 0.5 / b.y;

    let d = sqrt(max(b.y * b.y - a.y * p12.y, 0.0));
    var t1 = (b.y - d) * ra;
    var t2 = (b.y + d) * ra;

    // If polynomial is nearly linear, solve -2b*t + c = 0.
    if (abs(a.y) < 1.0 / 65536.0) {
        let t = p12.y * rb;
        t1 = t;
        t2 = t;
    }

    return vec2<f32>(
        (a.x * t1 - b.x * 2.0) * t1 + p12.x,
        (a.x * t2 - b.x * 2.0) * t2 + p12.x
    );
}

// Solve for y-coordinates where the curve crosses x = 0.
fn solveVertPoly(p12: vec4<f32>, p3: vec2<f32>) -> vec2<f32> {
    let a = vec2<f32>(p12.x - p12.z * 2.0 + p3.x, p12.y - p12.w * 2.0 + p3.y);
    let b = vec2<f32>(p12.x - p12.z, p12.y - p12.w);
    let ra = 1.0 / a.x;
    let rb = 0.5 / b.x;

    let d = sqrt(max(b.x * b.x - a.x * p12.x, 0.0));
    var t1 = (b.x - d) * ra;
    var t2 = (b.x + d) * ra;

    if (abs(a.x) < 1.0 / 65536.0) {
        let t = p12.x * rb;
        t1 = t;
        t2 = t;
    }

    return vec2<f32>(
        (a.y * t1 - b.y * 2.0) * t1 + p12.y,
        (a.y * t2 - b.y * 2.0) * t2 + p12.y
    );
}

// Compute 2D texture coordinate from glyph location + linear offset,
// wrapping at the band texture width (4096).
fn calcBandLoc(glyphLoc: vec2<i32>, offset: u32) -> vec2<i32> {
    var loc = vec2<i32>(glyphLoc.x + i32(offset), glyphLoc.y);
    loc.y += loc.x >> i32(BAND_TEX_LOG_W);
    loc.x &= (1 << i32(BAND_TEX_LOG_W)) - 1;
    return loc;
}

// Combine horizontal and vertical coverage using confidence weights.
fn calcCoverage(xcov: f32, ycov: f32, xwgt: f32, ywgt: f32) -> f32 {
    let coverage = max(
        abs(xcov * xwgt + ycov * ywgt) / max(xwgt + ywgt, 1.0 / 65536.0),
        min(abs(xcov), abs(ycov))
    );
    return clamp(coverage, 0.0, 1.0);
}

// --- Main Fragment Shader ---

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let renderCoord = in.texcoord;
    let bandTransform = in.banding;
    let glyphData = in.glyph;

    // Compute pixel-to-em-space ratio from screen-space derivatives.
    let emsPerPixel = fwidth(renderCoord);
    let pixelsPerEm = 1.0 / emsPerPixel;

    // Extract band maximums. bandMaxY is masked to lower 8 bits (upper bits = flags).
    let bandMaxX = glyphData.z;
    let bandMaxY = glyphData.w & 0xFF;

    // Map em-space coordinates to band indices.
    let bandIndex = clamp(
        vec2<i32>(renderCoord * bandTransform.xy + bandTransform.zw),
        vec2<i32>(0, 0),
        vec2<i32>(bandMaxX, bandMaxY)
    );
    let glyphLoc = glyphData.xy;

    // ---- Process horizontal band (rightward ray) ----

    var xcov: f32 = 0.0;
    var xwgt: f32 = 0.0;

    let hbandRaw = textureLoad(bandTexture, vec2<i32>(glyphLoc.x + bandIndex.y, glyphLoc.y), 0).xy;
    let hcount = min(hbandRaw.x, 256u);
    let hbandLoc = calcBandLoc(glyphLoc, hbandRaw.y);

    for (var ci: i32 = 0; ci < i32(hcount); ci++) {
        // Fetch curve location from the band's curve list.
        let rawLoc = textureLoad(bandTexture, vec2<i32>(hbandLoc.x + ci, hbandLoc.y), 0).xy;
        let curveLoc = vec2<i32>(i32(rawLoc.x), i32(rawLoc.y));

        // Fetch the three control points. Subtract renderCoord to make curve sample-relative.
        let p12 = textureLoad(curveTexture, curveLoc, 0) - vec4<f32>(renderCoord, renderCoord);
        let p3 = textureLoad(curveTexture, vec2<i32>(curveLoc.x + 1, curveLoc.y), 0).xy - renderCoord;

        // Early exit: if all control points are left of the pixel, no more curves can contribute.
        if (max(max(p12.x, p12.z), p3.x) * pixelsPerEm.x < -0.5) {
            break;
        }

        let code = calcRootCode(p12.y, p12.w, p3.y);
        if (code != 0u) {
            let r = solveHorizPoly(p12, p3) * pixelsPerEm.x;

            if ((code & 1u) != 0u) {
                xcov += clamp(r.x + 0.5, 0.0, 1.0);
                xwgt = max(xwgt, clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0));
            }
            if (code > 1u) {
                xcov -= clamp(r.y + 0.5, 0.0, 1.0);
                xwgt = max(xwgt, clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0));
            }
        }
    }

    // ---- Process vertical band (upward ray) ----

    var ycov: f32 = 0.0;
    var ywgt: f32 = 0.0;

    let vbandRaw = textureLoad(bandTexture, vec2<i32>(glyphLoc.x + bandMaxY + 1 + bandIndex.x, glyphLoc.y), 0).xy;
    let vcount = min(vbandRaw.x, 256u);
    let vbandLoc = calcBandLoc(glyphLoc, vbandRaw.y);

    for (var ci: i32 = 0; ci < i32(vcount); ci++) {
        let rawLoc = textureLoad(bandTexture, vec2<i32>(vbandLoc.x + ci, vbandLoc.y), 0).xy;
        let curveLoc = vec2<i32>(i32(rawLoc.x), i32(rawLoc.y));

        let p12 = textureLoad(curveTexture, curveLoc, 0) - vec4<f32>(renderCoord, renderCoord);
        let p3 = textureLoad(curveTexture, vec2<i32>(curveLoc.x + 1, curveLoc.y), 0).xy - renderCoord;

        // Early exit: if all control points are below the pixel.
        if (max(max(p12.y, p12.w), p3.y) * pixelsPerEm.y < -0.5) {
            break;
        }

        let code = calcRootCode(p12.x, p12.z, p3.x);
        if (code != 0u) {
            let r = solveVertPoly(p12, p3) * pixelsPerEm.y;

            if ((code & 1u) != 0u) {
                ycov -= clamp(r.x + 0.5, 0.0, 1.0);
                ywgt = max(ywgt, clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0));
            }
            if (code > 1u) {
                ycov += clamp(r.y + 0.5, 0.0, 1.0);
                ywgt = max(ywgt, clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0));
            }
        }
    }

    let coverage = calcCoverage(xcov, ycov, xwgt, ywgt);
    return in.color * coverage;
}
