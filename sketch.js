let video;
let handLandmarker;
let hands = [];
let smoothedHands = [];
let topologyGrid;

let modelReady = false;
let modelError = "";
let mirrorMode = false;
let lastVideoTime = -1;

const POINT_SIZE = 7;
const BONE_WEIGHT = 2.5;
const POSITION_SMOOTHING = 0.35;
const DEPTH_SMOOTHING = 0.25;
const ORIENTATION_SMOOTHING = 0.3;
const CLOSED_FIST_THRESHOLD = 0.28;
const GESTURE_HYSTERESIS = 0.08;

const WORLD_SCALE = 1300;
const WORLD_Z_SCALE = 1.2;
const GIZMO_SCALE = 80;
const INITIAL_CAMERA_DISTANCE = 1020;
const INITIAL_CAMERA_EYE_Y = -90;
const INITIAL_CAMERA_LOOK_Y = 130;


const DEPTH_SCALE = 2.0;
const HAND_SPAN_MIN = 120;
const HAND_SPAN_MAX = 360;
const GLOBAL_DEPTH_MIN = -400;
const GLOBAL_DEPTH_MAX = 400;
const MAX_HAND_INFLUENCE_POINTS = 42;
const GRID_Y_TRANSLATION = 100;

const HAND_PUSH_RADIUS = 150;
const HAND_PUSH_STRENGTH = 5;
const HAND_PUSH_FALLOFF = 0.5;
const HAND_PUSH_MAX = 200;

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17]
];
const FINGER_CHAINS = [
  [1, 2, 3, 4],    // thumb
  [5, 6, 7, 8],    // index
  [9, 10, 11, 12], // middle
  [13, 14, 15, 16], // ring
  [17, 18, 19, 20] // pinky
];

function setup() {
  createCanvas(windowWidth, windowHeight, WEBGL);
  // pixelDensity(1);
  camera(0, INITIAL_CAMERA_EYE_Y, -INITIAL_CAMERA_DISTANCE, 0, INITIAL_CAMERA_LOOK_Y, 0, 0, 1, 0);

  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();

  topologyGrid = new TopologyGrid({
    cols: 60,
    rows: 60,
    spacing: 28,
    circular: true,
    noiseScale: 0.09,
    noiseAmp: 100,
    noiseSpeed: 0.0025,
    yOffset: 250,
    handPushRadius: HAND_PUSH_RADIUS,
    handPushStrength: HAND_PUSH_STRENGTH,
    handPushFalloff: HAND_PUSH_FALLOFF,
    handPushMax: HAND_PUSH_MAX
  });

  if (typeof BOIDS3D_CONFIG !== "undefined" && BOIDS3D_CONFIG.scene) {
    BOIDS3D_CONFIG.scene.standaloneMode = false;
  }
  if (typeof initBoidsScene === "function") {
    initBoidsScene();
  }

  initHandLandmarker();
}

async function initHandLandmarker() {
  try {
    const vision = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14");
    const filesetResolver = await vision.FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );

    handLandmarker = await vision.HandLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
      },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.6,
      minHandPresenceConfidence: 0.6,
      minTrackingConfidence: 0.5
    });

    modelReady = true;
  } catch (error) {
    modelError = String(error);
  }
}

function draw() {
  let overlayMessage = "";

  background(10);
  orbitControl(1, 1, 1);
  lights();

  if (modelError) {
    overlayMessage = `Model load failed: ${modelError}`;
    hands = [];
    smoothedHands = [];
  } else if (!modelReady) {
    overlayMessage = "Loading MediaPipe Hand Landmarker...";
    hands = [];
    smoothedHands = [];
  } else {
    detectHands();
    updateSmoothedHands();
    if (hands.length === 0) {
      overlayMessage = "Show your hand to the webcam";
    }
  }

  const handInfluence = collectHandInfluenceXZ(smoothedHands);
  topologyGrid.setHandInfluence(handInfluence.points, handInfluence.count);

  push();
  translate(0, GRID_Y_TRANSLATION, 0);
  // TopologyGrid is already defined on the XZ plane, so no extra tilt is needed.
  topologyGrid.update();
  topologyGrid.draw();
  pop();

  updateBoidsHandInteraction(smoothedHands);
  if (typeof drawBoidsScene === "function") {
    drawBoidsScene();
  }

  // Re-apply scene lights after the custom shader pass to keep hand shading visible.
  lights();
  primeRendererForTopology();

  if (!overlayMessage) {
    for (const handState of smoothedHands) {
      drawHandSkeleton(handState.points, handState.gesture);
      drawOrientationGizmo(handState.orientation.center, handState.orientation);
    }

    drawOverlayDiagnostics();
  } else {
    drawOverlayText(overlayMessage);
  }
}

function updateBoidsHandInteraction(handStates) {
  if (typeof setBoidsHandControlActive !== "function" ||
      typeof setBoidsTargets !== "function") {
    return;
  }

  const activeHands = Array.isArray(handStates) ? handStates.slice(0, 2) : [];
  if (activeHands.length === 0) {
    setBoidsHandControlActive(false);
    setBoidsTargets([]);
    return;
  }

  const targets = activeHands.map((handState) => {
    const target = getBoidsTargetFromHand(handState);
    return {
      x: target.x,
      y: target.y,
      z: target.z,
      mode: handState.gesture === "closedFist" ? "repel" : "attractOrbit"
    };
  });
  setBoidsHandControlActive(true);
  setBoidsTargets(targets);
}

function getBoidsTargetFromHand(handState) {
  if (handState && handState.orientation && handState.orientation.center) {
    return handState.orientation.center;
  }

  const points = handState && Array.isArray(handState.points) ? handState.points : [];
  if (points.length === 0) {
    return createVector(0, 0, 0);
  }

  const sum = points.reduce((acc, point) => acc.add(point), createVector(0, 0, 0));
  return sum.div(points.length);
}

function collectHandInfluenceXZ(handStates) {
  const points = new Float32Array(MAX_HAND_INFLUENCE_POINTS * 2);
  let count = 0;

  for (const handState of handStates) {
    if (!handState || !Array.isArray(handState.points)) {
      continue;
    }

    for (const point of handState.points) {
      if (count >= MAX_HAND_INFLUENCE_POINTS) {
        return { points, count };
      }
      points[count * 2] = point.x;
      points[count * 2 + 1] = point.z;
      count += 1;
    }
  }

  return { points, count };
}

function detectHands() {
  if (!handLandmarker || !video || video.elt.readyState < 2) {
    return;
  }

  const currentTime = video.elt.currentTime;
  if (currentTime === lastVideoTime) {
    return;
  }
  lastVideoTime = currentTime;

  const result = handLandmarker.detectForVideo(video.elt, performance.now());
  const imageLandmarks = result.landmarks || [];
  const worldLandmarks = result.worldLandmarks || [];
  const handedness = result.handedness || [];

  hands = imageLandmarks.map((imagePoints, index) => {
    const handCategory = handedness[index] && handedness[index][0] ? handedness[index][0].categoryName : "Unknown";
    return {
      image: normalizeImageLandmarks(imagePoints),
      world: normalizeWorldLandmarks(worldLandmarks[index]),
      handedness: handCategory
    };
  });
}

function updateSmoothedHands() {
  const targets = hands
    .map((hand) => buildHandTarget(hand))
    .filter((target) => target.points.length === 21);

  if (targets.length !== smoothedHands.length) {
    smoothedHands = targets;
    return;
  }

  for (let handIndex = 0; handIndex < targets.length; handIndex += 1) {
    const target = targets[handIndex];
    const state = smoothedHands[handIndex];
    state.handedness = target.handedness;
    state.depthOffset = lerp(state.depthOffset, target.depthOffset, DEPTH_SMOOTHING);

    for (let pointIndex = 0; pointIndex < 21; pointIndex += 1) {
      const source = target.points[pointIndex];
      const current = state.points[pointIndex];
      current.x = lerp(current.x, source.x, POSITION_SMOOTHING);
      current.y = lerp(current.y, source.y, POSITION_SMOOTHING);
      current.z = lerp(current.z, source.z, POSITION_SMOOTHING);
    }

    state.orientation = smoothOrientation(state.orientation, target.orientation, ORIENTATION_SMOOTHING);
    state.rollDeg = lerp(state.rollDeg, target.rollDeg, ORIENTATION_SMOOTHING);
    state.pitchDeg = lerp(state.pitchDeg, target.pitchDeg, ORIENTATION_SMOOTHING);
    state.yawDeg = lerp(state.yawDeg, target.yawDeg, ORIENTATION_SMOOTHING);

    const gestureData = classifyHandGesture(target.points, state.gesture);
    state.gesture = gestureData.state;
    state.gestureScore = lerp(state.gestureScore, gestureData.score, ORIENTATION_SMOOTHING);
  }
}

function buildHandTarget(hand) {
  const depthOffset = estimateGlobalDepthOffset(hand.image);
  const useWorld = hand.world.length === 21;
  const points = useWorld
    ? buildWorldAnchoredPoints(hand.world, hand.image, depthOffset)
    : hand.image.map((point) => imageToScenePoint(point, depthOffset));

  const orientation = computeOrientationBasis(points, hand.handedness);
  const euler = orientationToEulerDegrees(orientation);
  const gestureData = classifyHandGesture(points);

  return {
    points,
    depthOffset,
    handedness: hand.handedness,
    orientation,
    rollDeg: euler.rollDeg,
    pitchDeg: euler.pitchDeg,
    yawDeg: euler.yawDeg,
    gesture: gestureData.state,
    gestureScore: gestureData.score
  };
}

function buildWorldAnchoredPoints(worldPoints, imagePoints, depthOffset) {
  const anchor = imagePalmCenterToScene(imagePoints, depthOffset);
  const worldCenter = averageWorldPoints(worldPoints);
  return worldPoints.map((point) => worldToScenePoint(point, worldCenter, anchor));
}

function worldToScenePoint(point, worldCenter, anchor) {
  const localX = (point.x - worldCenter.x) * WORLD_SCALE;
  const localY = (point.y - worldCenter.y) * WORLD_SCALE;
  const localZ = -(point.z - worldCenter.z) * WORLD_SCALE * WORLD_Z_SCALE;

  return createVector(
    anchor.x + (mirrorMode ? -localX : localX),
    anchor.y + localY,
    anchor.z + localZ
  );
}

function imageToScenePoint(point, depthOffset) {
  const coords = imageLandmarkToPixels(point);
  const px = mirrorMode ? video.width - coords.x : coords.x;
  const py = coords.y;
  const localDepth = -point[2] * DEPTH_SCALE;
  return createVector(
    map(px, 0, video.width, -width * 0.45, width * 0.45),
    map(py, 0, video.height, -height * 0.45, height * 0.45),
    localDepth + depthOffset
  );
}

function drawHandSkeleton(points, gesture) {
  
  let closedFistColor = {
    r: 208,
    g: 10,
    b: 100,
    a: 200 // opacity for closed fist (out of 255)
  }

  // Slightly brighter version of the topology lights color
  let openFistColor = {
    r: 255,
    g: 255,
    b: 255,
    a: 80 // opacity for open hand (out of 255)
  }

  let currentColor = gesture === "closedFist" ? closedFistColor : openFistColor;
  stroke(currentColor.r, currentColor.g, currentColor.b, currentColor.a);

  for (const point of points) {
    push();
    noFill();
    strokeWeight(0.5);
    stroke(currentColor.r, currentColor.g, currentColor.b, currentColor.a); // explicit per-point
    translate(point.x, point.y, point.z);
    sphere(POINT_SIZE, 8, 6);
    pop();
  }

  let coefficient = 0.3 // adjusts the cylinder width, didn't want to make a better name

  // Draw cylinders (bones) between hand joints instead of line segments
  noFill();
  for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
    const a = points[startIdx];
    const b = points[endIdx];
    const dir = p5.Vector.sub(b, a);
    const len = dir.mag();
    const mid = p5.Vector.add(a, b).mult(0.5);

    // Calculate rotation axis and angle
    const up = createVector(0, 1, 0);
    let axis = p5.Vector.cross(up, dir);
    let angle = acos(p5.Vector.dot(up.copy().normalize(), dir.copy().normalize()));
    if (axis.mag() < 0.0001) {
      axis = createVector(1, 0, 0); // Arbitrary axis
      if (dir.y > 0) angle = 0;
      else angle = PI;
    }
    push();
    translate(mid.x, mid.y, mid.z);
    // Apply orientation to align the cylinder with the bone
    rotate(angle, axis);
    ambientMaterial(currentColor.r, currentColor.g, currentColor.b, Math.round(currentColor.a * 0.7)); // semi-transparent bone color
    // noStroke();
    noFill();
    strokeWeight(0.2);
    // Use a slightly smaller radius than POINT_SIZE for more realistic bone thickness
    cylinder(POINT_SIZE * coefficient, len, 12, 1);
    pop();
  }
}

function drawOrientationGizmo(center, orientation) {
  const right = p5.Vector.mult(orientation.right, GIZMO_SCALE);
  const up = p5.Vector.mult(orientation.up, GIZMO_SCALE);
  const normal = p5.Vector.mult(orientation.normal, GIZMO_SCALE);

  // For some reason (unknown to man) when commenting this out the topology grid stops working
  // so instead I just made strokeWeight(0) and wont ask any more questions
  strokeWeight(0);
  stroke(255, 90, 90);
  line(center.x, center.y, center.z, center.x + right.x, center.y + right.y, center.z + right.z);
  stroke(90, 255, 120);
  line(center.x, center.y, center.z, center.x + up.x, center.y + up.y, center.z + up.z);
  stroke(90, 150, 255);
  line(center.x, center.y, center.z, center.x + normal.x, center.y + normal.y, center.z + normal.z);
}

function computeOrientationBasis(points, handLabel) {
  const wrist = points[0];
  const indexMcp = points[5];
  const middleMcp = points[9];
  const pinkyMcp = points[17];

  let right = p5.Vector.sub(indexMcp, pinkyMcp).normalize();
  let up = p5.Vector.sub(middleMcp, wrist).normalize();
  let normal = p5.Vector.cross(right, up).normalize();

  // Keep outward normal direction consistent between hands and mirror mode.
  const shouldFlip = (handLabel === "Left") !== mirrorMode;
  if (shouldFlip) {
    normal.mult(-1);
  }

  up = p5.Vector.cross(normal, right).normalize();
  right = p5.Vector.cross(up, normal).normalize();

  const center = averageVectors([points[0], points[5], points[9], points[13], points[17]]);

  return { center, right, up, normal };
}

function smoothOrientation(current, target, amount) {
  return {
    center: smoothVector(current.center, target.center, amount),
    right: smoothDirection(current.right, target.right, amount),
    up: smoothDirection(current.up, target.up, amount),
    normal: smoothDirection(current.normal, target.normal, amount)
  };
}

function smoothVector(a, b, amount) {
  return createVector(
    lerp(a.x, b.x, amount),
    lerp(a.y, b.y, amount),
    lerp(a.z, b.z, amount)
  );
}

function smoothDirection(a, b, amount) {
  const blended = createVector(
    lerp(a.x, b.x, amount),
    lerp(a.y, b.y, amount),
    lerp(a.z, b.z, amount)
  );
  if (blended.magSq() < 1e-8) {
    return b.copy();
  }
  return blended.normalize();
}

function orientationToEulerDegrees(orientation) {
  const m00 = orientation.right.x;
  const m10 = orientation.right.y;
  const m20 = orientation.right.z;
  const m21 = orientation.up.z;
  const m22 = orientation.normal.z;

  const pitch = asin(constrain(-m20, -1, 1));
  const roll = atan2(m21, m22);
  const yaw = atan2(m10, m00);

  return {
    rollDeg: degrees(roll),
    pitchDeg: degrees(pitch),
    yawDeg: degrees(yaw)
  };
}

function averageVectors(vectors) {
  const sum = vectors.reduce((acc, vec) => acc.add(vec), createVector(0, 0, 0));
  return sum.div(vectors.length);
}

function normalizeImageLandmarks(points) {
  if (!Array.isArray(points)) {
    return [];
  }

  return points.map((point) => [point.x, point.y, point.z || 0]);
}

function normalizeWorldLandmarks(points) {
  if (!Array.isArray(points)) {
    return [];
  }

  return points.map((point) => ({ x: point.x, y: point.y, z: point.z }));
}

function classifyHandGesture(points, previousGesture = "openPalm") {
  const score = computeHandOpennessScore(points);

  if (score <= CLOSED_FIST_THRESHOLD) {
    return { state: "closedFist", score };
  }

  if (previousGesture === "closedFist" && score <= CLOSED_FIST_THRESHOLD + GESTURE_HYSTERESIS) {
    return { state: "closedFist", score };
  }

  // If it's not confidently a fist, treat it as open palm by default.
  return { state: "openPalm", score };
}

function computeHandOpennessScore(points) {
  if (!Array.isArray(points) || points.length < 21) {
    return 0;
  }

  const wrist = points[0];
  const indexMcp = points[5];
  const middleMcp = points[9];
  const pinkyMcp = points[17];

  const palmScale = max(
    1,
    (p5.Vector.dist(wrist, middleMcp) + p5.Vector.dist(indexMcp, pinkyMcp)) * 0.5
  );

  let total = 0;
  for (let fingerIdx = 0; fingerIdx < FINGER_CHAINS.length; fingerIdx += 1) {
    const [mcpIdx, , pipIdx, tipIdx] = FINGER_CHAINS[fingerIdx];
    const mcp = points[mcpIdx];
    const pip = points[pipIdx];
    const tip = points[tipIdx];
    const reference = fingerIdx === 0 ? indexMcp : wrist;

    const tipDist = p5.Vector.dist(tip, reference);
    const pipDist = p5.Vector.dist(pip, reference);
    const extension = constrain((tipDist - pipDist) / (palmScale * 0.55), 0, 1);
    const straightness = constrain(p5.Vector.dist(tip, mcp) / (palmScale * 1.1), 0, 1);

    total += extension * 0.7 + straightness * 0.3;
  }

  return total / FINGER_CHAINS.length;
}

function estimateGlobalDepthOffset(imageLandmarks) {
  if (!Array.isArray(imageLandmarks) || imageLandmarks.length < 21) {
    return 0;
  }

  const wrist = imageLandmarkToPixels(imageLandmarks[0]);
  const middleMcp = imageLandmarkToPixels(imageLandmarks[9]);
  const indexMcp = imageLandmarkToPixels(imageLandmarks[5]);
  const pinkyMcp = imageLandmarkToPixels(imageLandmarks[17]);

  const span = dist(wrist.x, wrist.y, middleMcp.x, middleMcp.y) +
    dist(indexMcp.x, indexMcp.y, pinkyMcp.x, pinkyMcp.y);

  return map(span, HAND_SPAN_MIN, HAND_SPAN_MAX, GLOBAL_DEPTH_MIN, GLOBAL_DEPTH_MAX, true);
}

function imagePalmCenterToScene(imageLandmarks, depthOffset) {
  const palmIndices = [0, 5, 9, 13, 17];
  let sumX = 0;
  let sumY = 0;

  for (const index of palmIndices) {
    const point = imageLandmarkToPixels(imageLandmarks[index]);
    sumX += point.x;
    sumY += point.y;
  }

  const avgX = sumX / palmIndices.length;
  const avgY = sumY / palmIndices.length;
  const px = mirrorMode ? video.width - avgX : avgX;

  return createVector(
    map(px, 0, video.width, -width * 0.45, width * 0.45),
    map(avgY, 0, video.height, -height * 0.45, height * 0.45),
    depthOffset
  );
}

function averageWorldPoints(worldPoints) {
  const total = worldPoints.reduce(
    (acc, point) => {
      acc.x += point.x;
      acc.y += point.y;
      acc.z += point.z;
      return acc;
    },
    { x: 0, y: 0, z: 0 }
  );

  return {
    x: total.x / worldPoints.length,
    y: total.y / worldPoints.length,
    z: total.z / worldPoints.length
  };
}

function imageLandmarkToPixels(point) {
  const isNormalized = point[0] <= 1.5 && point[1] <= 1.5;
  return {
    x: isNormalized ? point[0] * video.width : point[0],
    y: isNormalized ? point[1] * video.height : point[1]
  };
}

function drawOverlayDiagnostics() {
  const firstHand = smoothedHands[0];
  const depthText = firstHand ? nf(firstHand.depthOffset, 1, 1) : "n/a";
  const handLabel = firstHand ? firstHand.handedness : "n/a";
  const gesture = firstHand ? firstHand.gesture : "n/a";
  const gestureScore = firstHand ? nf(firstHand.gestureScore, 1, 2) : "n/a";
  const roll = firstHand ? nf(firstHand.rollDeg, 1, 1) : "n/a";
  const pitch = firstHand ? nf(firstHand.pitchDeg, 1, 1) : "n/a";
  const yaw = firstHand ? nf(firstHand.yawDeg, 1, 1) : "n/a";

  drawOverlayText(
    `Hands: ${hands.length} | Hand: ${handLabel} | Gesture: ${gesture} (${gestureScore}) | Z offset: ${depthText}\n` +
    `Default: Open palm  Closed <= ${CLOSED_FIST_THRESHOLD} | ` +
    `Roll: ${roll}  Pitch: ${pitch}  Yaw: ${yaw} | Press M to mirror: ${mirrorMode ? "ON" : "OFF"}`
  );
}

function drawOverlayText(message) {
  push();
  resetMatrix();
  translate(-width / 2 + 16, -height / 2 + 16);
  noStroke();
  fill(255);
  textSize(16);
  textAlign(LEFT, TOP);
  text(message, 0, 0);
  pop();
}

function primeRendererForTopology() {
  // Keep this draw-state primer active even when no hands are visible.
  strokeWeight(0);
  stroke(255, 90, 90);
  line(0, 0, 0, 0, 0, 0);
}

function keyPressed() {
  if (key === "m" || key === "M") {
    mirrorMode = !mirrorMode;
  }
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}
