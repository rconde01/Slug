# Slug WebGPU Text Rendering - Development Log

## Overview

This project implements a WebGPU demo of the [Slug GPU text rendering algorithm](https://sluglibrary.com/) by Eric Lengyel. The algorithm renders text directly from quadratic Bezier curve outlines on the GPU, producing perfectly sharp, analytically anti-aliased text at any scale.

## Algorithm Summary

The Slug algorithm works by:

1. **Glyph representation**: Each glyph's outline is stored as a set of quadratic Bezier curves.
2. **Spatial subdivision**: The glyph's bounding box is divided into horizontal and vertical bands. Each band stores a list of curves that intersect it.
3. **GPU rendering**: Each glyph is drawn as a bounding quad. The fragment shader:
   - Determines which bands the current pixel falls in
   - Loops over curves in those bands
   - Casts two perpendicular rays (horizontal and vertical) through the pixel
   - Counts curve crossings to determine winding number / coverage
   - Combines horizontal and vertical coverage using confidence weights
   - Outputs `color * coverage` for premultiplied alpha blending

### Key Data Structures

- **Curve Texture** (`rgba32float`): Stores Bezier control points. Each curve occupies 2 texels: `(p1.x, p1.y, p2.x, p2.y)` and `(p3.x, p3.y, 0, 0)`.
- **Band Texture** (`rgba32uint`): Stores band headers and curve index lists. Each glyph occupies a contiguous block starting at `glyphLoc`.

## Architecture

```
index.html      - Page structure, CSS, script loading
app.js          - All application logic (ES module)
shaders.wgsl    - WGSL shaders (ported from HLSL reference)
DEVLOG.md       - This file
```

### Dependencies

- [opentype.js](https://opentype.js.org/) (CDN) - Font parsing
- WebGPU-capable browser (Chrome 113+, Edge 113+, Firefox Nightly)

## Phase 1: Initial Implementation

### Shader Port (HLSL → WGSL)

Ported the reference HLSL shaders to WGSL with these key translations:

| HLSL | WGSL |
|------|------|
| `asuint(x)` | `bitcast<u32>(x)` |
| `saturate(x)` | `clamp(x, 0.0, 1.0)` |
| `fwidth(x)` | `fwidth(x)` |
| `Texture2D.Load(int3(y,0))` | `textureLoad(tex, vec2<i32>(...), 0)` |
| `nointerpolation` | `@interpolate(flat)` |

**Simplification**: The vertex shader skips dynamic dilation (`SlugDilate`). Instead, CPU-side bounding box padding provides the extra margin needed for anti-aliasing at glyph edges.

### Font Processing

Using opentype.js to extract glyph outlines:
- TrueType fonts provide quadratic Bezier curves directly (`Q` commands)
- Line segments (`L` commands) converted to degenerate quadratics
- Cubic Bezier curves (`C` commands from CFF/OpenType) approximated with 2 quadratics via midpoint subdivision

### Band Data Building

For each unique glyph:
1. Extract quadratic curves from the glyph outline
2. Compute bounding box with padding
3. Choose band count: `clamp(ceil(sqrt(numCurves)), 2, 16)`
4. Assign curves to bands based on coordinate range overlap
5. Sort horizontal bands by descending max x, vertical by descending max y
6. Pack into curve and band textures

Each glyph gets its own row in the band texture to avoid wrapping complexities.

### Rendering Pipeline

- Premultiplied alpha blending: `src=One, dst=OneMinusSrcAlpha`
- No depth testing (all text in the same plane)
- No backface culling (visible from both sides when rotated)
- Perspective projection with arcball trackball rotation

## How to Run

Serve the directory with any HTTP server (needed for ES module and fetch):

```bash
# Python
python -m http.server 8000

# Node.js
npx serve .

# PHP
php -S localhost:8000
```

Then open `http://localhost:8000` in a WebGPU-capable browser.

## Controls

- **Text input**: Type text to render in the input field
- **Font file**: Load a .ttf/.otf/.woff font file
- **Color picker**: Change text color
- **Mouse drag**: Rotate text (trackball)
- **Scroll wheel**: Zoom in/out
