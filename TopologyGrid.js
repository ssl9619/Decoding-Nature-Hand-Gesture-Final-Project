const TOPOLOGY_VERT_SRC = `
precision highp float;

attribute vec2 aColRow;

uniform mat4 uModelViewMatrix;
uniform mat4 uProjectionMatrix;

uniform float uTime;
uniform float uSpacing;
uniform float uCols;
uniform float uRows;
uniform float uNoiseScale;
uniform float uNoiseAmp;
uniform float uYOffset;
uniform float uHandPointCount;
uniform float uHandPushRadius;
uniform float uHandPushStrength;
uniform float uHandPushFalloff;
uniform float uHandPushMax;
uniform vec2 uHandPointsXZ[42];

varying float vT;

// Simplex 3D noise (Ashima / Stefan Gustavson).
vec4 permute(vec4 x){ return mod(((x * 34.0) + 1.0) * x, 289.0); }
vec4 taylorInvSqrt(vec4 r){ return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
  const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

  vec3 i  = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);

  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);

  vec3 x1 = x0 - i1 + 1.0 * C.xxx;
  vec3 x2 = x0 - i2 + 2.0 * C.xxx;
  vec3 x3 = x0 - 1.0 + 3.0 * C.xxx;

  i = mod(i, 289.0);
  vec4 p = permute(permute(permute(
              i.z + vec4(0.0, i1.z, i2.z, 1.0))
            + i.y + vec4(0.0, i1.y, i2.y, 1.0))
            + i.x + vec4(0.0, i1.x, i2.x, 1.0));

  float n_ = 1.0 / 7.0;
  vec3 ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);

  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);

  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);

  vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

  vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

void main() {
  float col = aColRow.x;
  float row = aColRow.y;

  float gridW = (uCols - 1.0) * uSpacing;
  float gridD = (uRows - 1.0) * uSpacing;

  // Remap simplex output [-1,1] to [0,1] so colour/size mappings mirror the
  // original p5 noise() pipeline.
  float n = snoise(vec3(col * uNoiseScale, row * uNoiseScale, uTime)) * 0.5 + 0.5;

  float x = col * uSpacing - gridW * 0.5;
  float z = row * uSpacing - gridD * 0.5;
  float baseY = -n * uNoiseAmp + uYOffset;
  vec2 pointXZ = vec2(x, z);

  float pushAmount = 0.0;
  for (int i = 0; i < 42; i++) {
    if (float(i) >= uHandPointCount) {
      break;
    }
    float d = distance(pointXZ, uHandPointsXZ[i]);
    float radiusT = clamp(1.0 - (d / max(uHandPushRadius, 0.0001)), 0.0, 1.0);
    float influence = pow(radiusT, max(uHandPushFalloff, 0.0001));
    pushAmount += influence * uHandPushStrength;
  }

  pushAmount = min(pushAmount, uHandPushMax);
  float y = baseY + pushAmount;

  vT = n;

  gl_Position = uProjectionMatrix * uModelViewMatrix * vec4(x, y, z, 1.0);
  gl_PointSize = mix(1.5, 5.0, n);
}
`;

const TOPOLOGY_FRAG_SRC = `
precision highp float;

varying float vT;

void main() {
  vec3 color = mix(vec3(60.0, 30.0, 10.0), vec3(255.0, 190.0, 100.0), vT) / 255.0;
  float alpha = mix(80.0, 255.0, vT) / 255.0;
  gl_FragColor = vec4(color, alpha);
}
`;

const TOPOLOGY_MAX_HAND_POINTS = 42;

class TopologyGrid {
  constructor({
    cols       = 70,
    rows       = 70,
    spacing    = 28,
    circular   = true,
    noiseScale = 0.09,
    noiseAmp   = 180,
    noiseSpeed = 0.0025,
    yOffset    = 100,
    handPushRadius = 240,
    handPushStrength = 24,
    handPushFalloff = 1.75,
    handPushMax = 140,
  } = {}) {
    this.cols       = cols;
    this.rows       = rows;
    this.spacing    = spacing;
    this.circular   = circular;
    this.noiseScale = noiseScale;
    this.noiseAmp   = noiseAmp;
    this.noiseSpeed = noiseSpeed;
    this.yOffset    = yOffset;
    this.handPushRadius = handPushRadius;
    this.handPushStrength = handPushStrength;
    this.handPushFalloff = handPushFalloff;
    this.handPushMax = handPushMax;
    this.maxInfluencePoints = TOPOLOGY_MAX_HAND_POINTS;
    this._noiseZ    = 0;
    this._initialized = false;
    this._handPointCount = 0;
    this._handPointsXZ = new Float32Array(this.maxInfluencePoints * 2);
    this._handPointsUniform = Array.from(this._handPointsXZ);
  }

  setHandInfluence(pointsXZ, activeCount) {
    const clampedCount = Math.max(0, Math.min(this.maxInfluencePoints, activeCount | 0));
    this._handPointCount = clampedCount;
    this._handPointsXZ.fill(0);

    if (pointsXZ && pointsXZ.length) {
      const maxValues = clampedCount * 2;
      const copyLength = Math.min(maxValues, pointsXZ.length);
      for (let i = 0; i < copyLength; i += 1) {
        this._handPointsXZ[i] = pointsXZ[i];
      }
    }

    this._handPointsUniform = Array.from(this._handPointsXZ);
  }

  _initGPU() {
    const gl = drawingContext;
    this._gl = gl;

    this._shader = createShader(TOPOLOGY_VERT_SRC, TOPOLOGY_FRAG_SRC);
    this._shader.init();
    this._aColRowLoc = gl.getAttribLocation(this._shader._glProgram, "aColRow");

    const gridW = (this.cols - 1) * this.spacing;
    const gridD = (this.rows - 1) * this.spacing;
    const radius = Math.min(gridW, gridD) * 0.5;
    const radiusSq = radius * radius;
    const rawData = [];

    for (let row = 0; row < this.rows; row++) {
      for (let col = 0; col < this.cols; col++) {
        const x = col * this.spacing - gridW * 0.5;
        const z = row * this.spacing - gridD * 0.5;
        const insideCircle = !this.circular || ((x * x) + (z * z) <= radiusSq);
        if (!insideCircle) {
          continue;
        }

        rawData.push(col, row);
      }
    }
    const data = new Float32Array(rawData);

    this._buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this._buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    this._vertexCount = data.length / 2;
    this._initialized = true;
  }

  update() {
    this._noiseZ += this.noiseSpeed;
  }

  draw() {
    if (!this._initialized) {
      this._initGPU();
    }

    const gl = this._gl;
    const sh = this._shader;

    shader(sh);

    sh.setUniform("uTime", this._noiseZ);
    sh.setUniform("uSpacing", this.spacing);
    sh.setUniform("uCols", this.cols);
    sh.setUniform("uRows", this.rows);
    sh.setUniform("uNoiseScale", this.noiseScale);
    sh.setUniform("uNoiseAmp", this.noiseAmp);
    sh.setUniform("uYOffset", this.yOffset);
    sh.setUniform("uHandPointCount", this._handPointCount);
    sh.setUniform("uHandPushRadius", this.handPushRadius);
    sh.setUniform("uHandPushStrength", this.handPushStrength);
    sh.setUniform("uHandPushFalloff", this.handPushFalloff);
    sh.setUniform("uHandPushMax", this.handPushMax);
    sh.setUniform("uHandPointsXZ", this._handPointsUniform);
    sh.setUniform("uModelViewMatrix", _renderer.uMVMatrix.mat4);
    sh.setUniform("uProjectionMatrix", _renderer.uPMatrix.mat4);

    gl.bindBuffer(gl.ARRAY_BUFFER, this._buffer);
    if (this._aColRowLoc >= 0) {
      gl.enableVertexAttribArray(this._aColRowLoc);
      gl.vertexAttribPointer(this._aColRowLoc, 2, gl.FLOAT, false, 0, 0);
    }

    gl.drawArrays(gl.POINTS, 0, this._vertexCount);

    // Avoid leaking VBO binding into p5's subsequent draw calls.
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    resetShader();
  }
}
