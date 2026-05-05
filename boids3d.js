const BOIDS3D_CONFIG = {
  population: {
    boidCount: 300,
  },

  scene: {
    standaloneMode: true,     // set false when embedded in another sketch
    worldHalfExtent: 520,
    worldHeight: 320,
    floorY: 180,
    grid: {
      halfSize: 1000,
      spacing: 26,
      waveAmplitude: 5.5,
    },
  },

  motion: {
    maxSpeed: 2.8,
    maxForce: 0.09,
    drag: 0.985,
    wanderWeight: 0.16,
  },

  flocking: {
    radii: {
      separation: 28,
      alignment: 58,
      cohesion: 70,
    },
    weights: {
      separation: 1.4,
      alignment: 0.65,
      cohesion: 0.45,
    },
  },

  interaction: {
    repel: {
      strength: 56,
      range: 260,
      escapeRepelMultiplier: 1.45,
    },
    orbit: {
      radius: 95,
      radialGain: 0.05,
      tangentialGain: 0.2,
      verticalGain: 0.02,
    },
    targetMotion: {
      amplitudeX: 170,
      amplitudeY: 95,
      amplitudeZ: 170,
    },
  },

  network: {
    maxNeighborConnections: 5,
    connectionStrokeWeight: 1.15,
    connectionAlpha: 36,
  },

  state: {
    followBeforeWanderFrames: 260,
    baseWanderChancePerFrame: 0.0012,
    neighborWanderChanceBonus: 0.0022,
    maxWanderChancePerFrame: 0.03,
    wanderDurationMinFrames: 110,
    wanderDurationMaxFrames: 230,
    wanderCooldownFrames: 190,
    followWanderWeight: 0.04,
    wanderStateWanderWeight: 0.22,
    escapeWanderWeight: 0.1,
    escapeTriggerDistance: 190,
    escapeToWanderDistance: 320,
  },

  visual: {
    boidSize: 3.2,
    wanderVisualScale: 1.2,
    escapeVisualScale: 1.12,
    boidStrokeWeight: 0.95,
    boidSizeMultiplier: 2.1,
    followVisualAlpha: 165,
    wanderVisualAlpha: 180,
    escapeVisualAlpha: 170,
    followColor: [190, 230, 255],
    wanderColor: [255, 240, 185],
    escapeColor: [255, 170, 200],
  },
};

let boidsScene = null;

function initBoidsScene() {
  boidsScene = new BoidsScene();
}

function drawBoidsScene() {
  if (!boidsScene) {
    return;
  }
  boidsScene.stepAndRender();
}

function setBoidsInteractionMode(mode) {
  if (!boidsScene) {
    return;
  }
  if (mode !== "repel" && mode !== "attractOrbit") {
    return;
  }
  boidsScene.mode = mode;
  boidsScene.targetModes[0] = mode;
  boidsScene.targetModes[1] = mode;
}

// Set the boids target position directly; call every frame while hand is present.
function setBoidsTargetPosition(x, y, z) {
  if (!boidsScene) return;
  boidsScene.target.set(x, y, z);
  boidsScene.targets[0].set(x, y, z);
  boidsScene.activeTargetCount = max(boidsScene.activeTargetCount, 1);
}

// Pass false to freeze auto target motion; true to re-enable it.
function setBoidsTargetMotionEnabled(enabled) {
  if (!boidsScene) return;
  boidsScene.externalTargetEnabled = !enabled;
  if (enabled) {
    boidsScene.handControlActive = false;
  }
}

// Returns the current interaction mode string ("attractOrbit" | "repel").
function getBoidsInteractionMode() {
  return boidsScene ? boidsScene.mode : null;
}

// Enables/disables explicit hand-driven boid control.
function setBoidsHandControlActive(active) {
  if (!boidsScene) return;
  boidsScene.handControlActive = Boolean(active);
}

// Sets up to two boid targets with per-target mode.
function setBoidsTargets(targetSpecs) {
  if (!boidsScene || !Array.isArray(targetSpecs)) {
    return;
  }

  const maxTargets = min(2, targetSpecs.length);
  boidsScene.activeTargetCount = maxTargets;
  for (let i = 0; i < maxTargets; i += 1) {
    const spec = targetSpecs[i] || {};
    const x = Number(spec.x) || 0;
    const y = Number(spec.y) || 0;
    const z = Number(spec.z) || 0;
    const mode = spec.mode === "repel" ? "repel" : "attractOrbit";
    boidsScene.targets[i].set(x, y, z);
    boidsScene.targetModes[i] = mode;
  }

  if (maxTargets > 0) {
    boidsScene.target.set(
      boidsScene.targets[0].x,
      boidsScene.targets[0].y,
      boidsScene.targets[0].z
    );
    boidsScene.mode = boidsScene.targetModes[0];
  }
}

class BoidsScene {
  constructor() {
    this.config = BOIDS3D_CONFIG;
    this.mode = "attractOrbit";
    this.target = createVector(0, 0, 0);
    this.externalTargetEnabled = false;
    this.handControlActive = false;
    this.activeTargetCount = 0;
    this.targets = [createVector(0, 0, 0), createVector(0, 0, 0)];
    this.targetModes = ["attractOrbit", "attractOrbit"];
    this.boids = [];
    this.neighborGraph = [];

    for (let i = 0; i < this.config.population.boidCount; i += 1) {
      this.boids.push(new Boid3D(this.config));
    }
  }

  stepAndRender() {
    const standalone = this.config.scene.standaloneMode !== false;
    if (standalone) {
      background(0);
      orbitControl(1.15, 1.15, 0.12);
    }

    const hasHandTargets = this.handControlActive && this.activeTargetCount > 0;
    if (!hasHandTargets && !this.externalTargetEnabled) {
      this.updateTarget();
    }
    this.computeNeighborGraph();
    if (standalone) {
      this.drawFloorGrid();
    }
    this.drawTargets(hasHandTargets);

    const boidModes = new Array(this.boids.length);
    for (let i = 0; i < this.boids.length; i += 1) {
      const boid = this.boids[i];
      const connectedWanderCount = this.countWanderingNeighbors(i);
      boid.flock(this.boids);
      boid.applyContainment();

      if (!hasHandTargets) {
        if (boid.state !== "wander") {
          boid.enterWanderState();
        }
        boidModes[i] = "attractOrbit";
        boid.applyWander(this.config.state.wanderStateWanderWeight);
      } else {
        const targetInfo = this.resolveTargetForBoid(boid);
        const target = targetInfo.target;
        const mode = targetInfo.mode;
        boidModes[i] = mode;
        boid.updateState(connectedWanderCount, mode, target);

        if (boid.state === "escape") {
          boid.applyWander(this.config.state.escapeWanderWeight);
          boid.applyRepel(target, this.config.interaction.repel.escapeRepelMultiplier);
        } else if (boid.state === "wander") {
          boid.applyWander(this.config.state.wanderStateWanderWeight);
        } else {
          boid.applyWander(this.config.state.followWanderWeight);
          if (mode === "repel") {
            boid.applyRepel(target);
          } else {
            boid.applyAttractOrbit(target);
          }
        }
      }

      boid.integrate();
    }

    this.drawNeighborConnections(boidModes);

    for (let i = 0; i < this.boids.length; i += 1) {
      this.boids[i].render(boidModes[i]);
    }
  }

  updateTarget() {
    const targetMotionConfig = this.config.interaction.targetMotion;
    const t = frameCount * 0.0135;
    this.target.set(
      sin(t) * targetMotionConfig.amplitudeX,
      sin(t * 1.7) * targetMotionConfig.amplitudeY,
      cos(t * 0.84) * targetMotionConfig.amplitudeZ
    );
  }

  drawTargets(hasHandTargets) {
    if (hasHandTargets) {
      for (let i = 0; i < this.activeTargetCount; i += 1) {
        this.drawSingleTarget(this.targets[i], this.targetModes[i]);
      }
      return;
    }

    if (!this.externalTargetEnabled) {
      this.drawSingleTarget(this.target, this.mode);
    }
  }

  drawSingleTarget(target, mode) {
    // Target marker intentionally hidden.
  }

  drawFloorGrid() {
    const config = this.config;
    const gridConfig = config.scene.grid;
    strokeWeight(1.6);

    for (let x = -gridConfig.halfSize; x <= gridConfig.halfSize; x += gridConfig.spacing) {
      for (let z = -gridConfig.halfSize; z <= gridConfig.halfSize; z += gridConfig.spacing) {
        const d = dist(x, z, 0, 0);
        if (d > gridConfig.halfSize) {
          continue;
        }

        const edgeFade = 1 - pow(d / gridConfig.halfSize, 1.35);
        if (edgeFade <= 0.025) {
          continue;
        }

        const wave = sin((x + frameCount * 0.9) * 0.012) + cos((z - frameCount * 0.7) * 0.013);
        const y = config.scene.floorY + wave * gridConfig.waveAmplitude;
        stroke(225, 180, 120, 12 + edgeFade * 88);
        point(x, y, z);
      }
    }
  }

  computeNeighborGraph() {
    const linkCount = this.config.network.maxNeighborConnections;
    this.neighborGraph = new Array(this.boids.length);

    if (linkCount <= 0) {
      for (let i = 0; i < this.boids.length; i += 1) {
        this.neighborGraph[i] = [];
      }
      return;
    }

    for (let i = 0; i < this.boids.length; i += 1) {
      const boid = this.boids[i];
      const distances = [];

      for (let j = 0; j < this.boids.length; j += 1) {
        if (i === j) {
          continue;
        }
        const other = this.boids[j];
        distances.push({
          idx: j,
          dSq: p5.Vector.sub(other.pos, boid.pos).magSq(),
        });
      }

      distances.sort((a, b) => a.dSq - b.dSq);
      const nearest = [];
      const localCount = min(linkCount, distances.length);
      for (let k = 0; k < localCount; k += 1) {
        nearest.push(distances[k].idx);
      }
      this.neighborGraph[i] = nearest;
    }
  }

  countWanderingNeighbors(boidIndex) {
    const neighbors = this.neighborGraph[boidIndex] || [];
    let count = 0;

    for (const idx of neighbors) {
      if (this.boids[idx].state === "wander") {
        count += 1;
      }
    }

    return count;
  }

  resolveTargetForBoid(boid) {
    if (this.activeTargetCount <= 1) {
      return {
        target: this.targets[0],
        mode: this.targetModes[0],
      };
    }

    const dSq0 = p5.Vector.sub(this.targets[0], boid.pos).magSq();
    const dSq1 = p5.Vector.sub(this.targets[1], boid.pos).magSq();
    const index = dSq0 <= dSq1 ? 0 : 1;
    return {
      target: this.targets[index],
      mode: this.targetModes[index],
    };
  }

  getBoidVisualStyle(boid, mode = "attractOrbit") {
    const visual = this.config.visual;
    const isWander = boid.state === "wander";
    const isEscape = boid.state === "escape";
    if (isEscape) {
      return {
        color: visual.escapeColor,
        alpha: visual.escapeVisualAlpha,
        scale: visual.escapeVisualScale,
      };
    }
    if (isWander) {
      return {
        color: visual.wanderColor,
        alpha: visual.wanderVisualAlpha,
        scale: visual.wanderVisualScale,
      };
    }
    return {
      color: visual.followColor,
      alpha: visual.followVisualAlpha,
      scale: 1,
    };
  }

  drawNeighborConnections(boidModes) {
    const config = this.config;
    if (config.network.maxNeighborConnections <= 0) {
      return;
    }

    strokeWeight(config.network.connectionStrokeWeight);

    for (let i = 0; i < this.boids.length; i += 1) {
      const boid = this.boids[i];
      const neighbors = this.neighborGraph[i] || [];
      const boidStyle = this.getBoidVisualStyle(boid, boidModes[i]);
      stroke(
        boidStyle.color[0],
        boidStyle.color[1],
        boidStyle.color[2],
        config.network.connectionAlpha
      );

      for (const idx of neighbors) {
        const neighbor = this.boids[idx];
        line(
          boid.pos.x,
          boid.pos.y,
          boid.pos.z,
          neighbor.pos.x,
          neighbor.pos.y,
          neighbor.pos.z
        );
      }
    }
  }
}

class Boid3D {
  constructor(config) {
    this.config = config;
    const sceneConfig = config.scene;
    const stateConfig = config.state;
    this.pos = createVector(
      random(-sceneConfig.worldHalfExtent * 0.45, sceneConfig.worldHalfExtent * 0.45),
      random(-sceneConfig.worldHeight * 0.45, sceneConfig.worldHeight * 0.25),
      random(-sceneConfig.worldHalfExtent * 0.45, sceneConfig.worldHalfExtent * 0.45)
    );
    this.vel = p5.Vector.random3D().mult(random(0.8, 1.8));
    this.acc = createVector(0, 0, 0);
    this.phase = random(TWO_PI);
    this.state = "follow";
    this.stateTimer = floor(random(0, stateConfig.followBeforeWanderFrames * 0.75));
    this.wanderCooldown = floor(random(0, stateConfig.wanderCooldownFrames * 0.8));
    this.wanderDurationLeft = 0;
  }

  flock(boids) {
    const sep = createVector(0, 0, 0);
    const ali = createVector(0, 0, 0);
    const coh = createVector(0, 0, 0);
    let sepCount = 0;
    let aliCount = 0;
    let cohCount = 0;

    for (const other of boids) {
      if (other === this) {
        continue;
      }

      const d = p5.Vector.dist(this.pos, other.pos);
      if (d <= 0.0001) {
        continue;
      }

      if (d < this.config.flocking.radii.separation) {
        const away = p5.Vector.sub(this.pos, other.pos).normalize().div(d);
        sep.add(away);
        sepCount += 1;
      }
      if (d < this.config.flocking.radii.alignment) {
        ali.add(other.vel);
        aliCount += 1;
      }
      if (d < this.config.flocking.radii.cohesion) {
        coh.add(other.pos);
        cohCount += 1;
      }
    }

    if (sepCount > 0) {
      sep.div(sepCount);
      sep.setMag(this.config.motion.maxSpeed);
      sep.sub(this.vel);
      sep.limit(this.config.motion.maxForce * this.config.flocking.weights.separation);
      this.acc.add(sep.mult(this.config.flocking.weights.separation));
    }

    if (aliCount > 0) {
      ali.div(aliCount);
      ali.setMag(this.config.motion.maxSpeed);
      ali.sub(this.vel);
      ali.limit(this.config.motion.maxForce * this.config.flocking.weights.alignment);
      this.acc.add(ali.mult(this.config.flocking.weights.alignment));
    }

    if (cohCount > 0) {
      coh.div(cohCount);
      const seek = p5.Vector.sub(coh, this.pos);
      seek.setMag(this.config.motion.maxSpeed);
      seek.sub(this.vel);
      seek.limit(this.config.motion.maxForce * this.config.flocking.weights.cohesion);
      this.acc.add(seek.mult(this.config.flocking.weights.cohesion));
    }
  }

  updateState(connectedWanderCount, mode, target) {
    const config = this.config;
    const stateConfig = config.state;
    const distToTarget = p5.Vector.dist(this.pos, target);

    if (this.wanderCooldown > 0) {
      this.wanderCooldown -= 1;
    }

    if (mode === "repel") {
      if (this.state !== "escape" && distToTarget < stateConfig.escapeTriggerDistance) {
        this.enterEscapeState();
      }
      if (this.state === "escape") {
        if (distToTarget >= stateConfig.escapeToWanderDistance) {
          this.enterWanderState();
        }
        return;
      }
    } else if (this.state === "escape") {
      this.enterFollowState();
    }

    if (this.state === "wander") {
      this.stateTimer += 1;
      this.wanderDurationLeft -= 1;
      if (this.wanderDurationLeft <= 0) {
        this.enterFollowState();
      }
      return;
    }

    this.stateTimer += 1;
    if (this.stateTimer < stateConfig.followBeforeWanderFrames || this.wanderCooldown > 0) {
      return;
    }

    const chance =
      stateConfig.baseWanderChancePerFrame +
      connectedWanderCount * stateConfig.neighborWanderChanceBonus;
    const cappedChance = min(chance, stateConfig.maxWanderChancePerFrame);
    if (random() < cappedChance) {
      this.enterWanderState();
    }
  }

  enterFollowState() {
    this.state = "follow";
    this.stateTimer = 0;
    this.wanderDurationLeft = 0;
    this.wanderCooldown = this.config.state.wanderCooldownFrames;
  }

  enterWanderState() {
    this.state = "wander";
    this.stateTimer = 0;
    const stateConfig = this.config.state;
    this.wanderDurationLeft = floor(
      random(stateConfig.wanderDurationMinFrames, stateConfig.wanderDurationMaxFrames + 1)
    );
  }

  enterEscapeState() {
    this.state = "escape";
    this.stateTimer = 0;
    this.wanderDurationLeft = 0;
  }

  applyWander(weight = this.config.motion.wanderWeight) {
    this.phase += random(-0.15, 0.15);
    const wander = createVector(
      cos(this.phase),
      sin(this.phase * 0.7) * 0.35,
      sin(this.phase)
    );
    wander.setMag(weight);
    this.acc.add(wander);
  }

  applyContainment() {
    const config = this.config;
    const sceneConfig = config.scene;
    const steer = createVector(0, 0, 0);

    if (this.pos.x < -sceneConfig.worldHalfExtent) {
      steer.x += 1;
    } else if (this.pos.x > sceneConfig.worldHalfExtent) {
      steer.x -= 1;
    }
    if (this.pos.y < -sceneConfig.worldHeight) {
      steer.y += 1;
    } else if (this.pos.y > sceneConfig.floorY - 8) {
      steer.y -= 1;
    }
    if (this.pos.z < -sceneConfig.worldHalfExtent) {
      steer.z += 1;
    } else if (this.pos.z > sceneConfig.worldHalfExtent) {
      steer.z -= 1;
    }

    if (steer.magSq() > 0) {
      steer.setMag(config.motion.maxForce * 2.5);
      this.acc.add(steer);
    }
  }

  applyRepel(target, multiplier = 1) {
    const repelConfig = this.config.interaction.repel;
    const offset = p5.Vector.sub(this.pos, target);
    const d = offset.mag() + 0.0001;
    if (d > repelConfig.range) {
      return;
    }

    const t = 1 - d / repelConfig.range;
    const strength = repelConfig.strength * multiplier * t * t;
    offset.normalize().mult(strength);
    offset.limit(this.config.motion.maxForce * 3.2);
    this.acc.add(offset);
  }

  applyAttractOrbit(target) {
    const radial = p5.Vector.sub(target, this.pos);
    const d = radial.mag() + 0.0001;
    const radialDir = radial.copy().div(d);

    const orbitConfig = this.config.interaction.orbit;
    const orbitError = d - orbitConfig.radius;
    const radialPull = radialDir.copy().mult(orbitError * orbitConfig.radialGain);

    const globalUp = createVector(0, 1, 0);
    let tangent = p5.Vector.cross(radialDir, globalUp);
    if (tangent.magSq() < 0.0001) {
      tangent = createVector(1, 0, 0);
    } else {
      tangent.normalize();
    }
    const tangential = tangent.mult(orbitConfig.tangentialGain);

    const verticalBias = (target.y - this.pos.y) * orbitConfig.verticalGain;
    const blend = createVector(
      radialPull.x + tangential.x,
      radialPull.y + verticalBias,
      radialPull.z + tangential.z
    );

    blend.limit(this.config.motion.maxForce * 2.4);
    this.acc.add(blend);
  }

  integrate() {
    this.vel.add(this.acc);
    this.vel.limit(this.config.motion.maxSpeed);
    this.vel.mult(this.config.motion.drag);
    this.pos.add(this.vel);
    this.acc.mult(0);
  }

  render(mode = "attractOrbit") {
    push();
    translate(this.pos.x, this.pos.y, this.pos.z);
    noFill();
    const visual = this.config.visual;
    const isWander = this.state === "wander";
    const isEscape = this.state === "escape";
    const alpha = isEscape
      ? visual.escapeVisualAlpha
      : isWander
        ? visual.wanderVisualAlpha
        : visual.followVisualAlpha;
    const scale = isEscape
      ? visual.escapeVisualScale
      : isWander
        ? visual.wanderVisualScale
        : 1;
    const stateColor = isEscape
      ? visual.escapeColor
      : isWander
        ? visual.wanderColor
        : visual.followColor;
    strokeWeight(visual.boidStrokeWeight);
    stroke(stateColor[0], stateColor[1], stateColor[2], alpha);
    circle(0, 0, visual.boidSize * visual.boidSizeMultiplier * scale);
    pop();
  }
}
