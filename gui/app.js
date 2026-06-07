const cases = {
  case01: "case01_test_3752-4943-0010_777-126732-0024",
  case02: "case02_test_3752-4944-0000_5694-64025-0004",
  case03: "case03_test_3752-4944-0011_1993-147149-0019",
};

const tracks = ["mixture", "ours", "baseline", "target"];
const palette = {
  cyan: "#62e3ff",
  green: "#8ff0b0",
  amber: "#ffcf77",
  red: "#ff7f86",
  muted: "#8ea0ae",
  line: "#263340",
  bg: "#080c10",
};
const caseProfiles = {
  case01: {
    seed: 101,
    title: "案例 01 · 清晰重叠",
    note: "目标与干扰有较清楚的停顿边界，目标提取输出应更接近绿色目标真值。",
    sdr: 10.59,
    sdri: 10.58,
    density: 0.92,
    peaks: [0.18, 0.34, 0.58, 0.77],
    rows: [
      { label: "混合语音", color: palette.amber, intensity: 1.04, noise: 0.45, bias: 0.12 },
      { label: "基线模型", color: palette.red, intensity: 0.76, noise: 0.28, bias: 0.23 },
      { label: "目标提取输出", color: palette.cyan, intensity: 0.62, noise: 0.16, bias: 0.05 },
      { label: "目标真值", color: palette.green, intensity: 0.58, noise: 0.10, bias: 0.02 },
    ],
  },
  case02: {
    seed: 207,
    title: "案例 02 · 同性别混合",
    note: "目标与干扰音色更接近，频谱重叠更密，参考语音条件负责拉开身份差异。",
    sdr: 10.80,
    sdri: 10.80,
    density: 1.18,
    peaks: [0.11, 0.25, 0.43, 0.62, 0.86],
    rows: [
      { label: "混合语音", color: palette.amber, intensity: 1.16, noise: 0.62, bias: 0.28 },
      { label: "基线模型", color: palette.red, intensity: 0.91, noise: 0.44, bias: 0.34 },
      { label: "目标提取输出", color: palette.cyan, intensity: 0.70, noise: 0.22, bias: 0.14 },
      { label: "目标真值", color: palette.green, intensity: 0.66, noise: 0.12, bias: 0.08 },
    ],
  },
  case03: {
    seed: 313,
    title: "案例 03 · 密集干扰",
    note: "干扰更连续，低频和中频能量更拥挤，错误参考语音会使目标选择明显崩溃。",
    sdr: 10.09,
    sdri: 10.08,
    density: 1.42,
    peaks: [0.08, 0.21, 0.37, 0.51, 0.68, 0.91],
    rows: [
      { label: "混合语音", color: palette.amber, intensity: 1.28, noise: 0.76, bias: 0.40 },
      { label: "基线模型", color: palette.red, intensity: 1.02, noise: 0.58, bias: 0.46 },
      { label: "目标提取输出", color: palette.cyan, intensity: 0.78, noise: 0.30, bias: 0.21 },
      { label: "目标真值", color: palette.green, intensity: 0.70, noise: 0.14, bias: 0.12 },
    ],
  },
};
const baseline = {
  sdr: 10.34,
  sisdr: 10.32,
  params: 0,
  macs: 0,
  rtf: 1.00,
};
const finalSystem = {
  sdr: 10.799,
  sisdr: 10.804,
  params: 0,
  macs: 0,
  rtf: 1.0,
};
const moduleInfo = {
  mixture: {
    title: "混合语音输入",
    body: "输入端模拟鸡尾酒会场景：2-3 个说话人同时发声，并可叠加背景噪声。它对应真实听觉场景中的复杂声学流，是系统必须先解析的对象。",
  },
  stft: {
    title: "时域编码器",
    body: "TD-SpeakerBeam 使用可学习时域滤波器把一维波形编码为适合掩码估计的表示，避免显式相位估计。",
  },
  egsp: {
    title: "EGSP 频域探针",
    body: "EGSP 是认知启发式探针：根据参考语音频谱给 mixture 加权。实验表明推理时硬加会降低指标，因此报告把它作为负结果和训练一致性反思。",
  },
  condition: {
    title: "说话人条件融合",
    body: "参考语音不仅是标签，而是目标身份线索。系统把参考语音转为说话人条件，并注入分离主干，使模型知道要保留哪一个说话人的声音。",
  },
  separator: {
    title: "TD-SpeakerBeam 主干",
    body: "分离主干接收 mixture 表示和目标 speaker embedding，输出目标说话人的估计波形。可靠主干把系统从早期 1 dB 失败原型提升到 10 dB 以上。",
  },
  anchor: {
    title: "多参考 embedding pooling",
    body: "对同一说话人的多条参考语音分别提取 embedding，再在 embedding 层平均。k=2 将 mid fine-tune 学生从 10.586 dB 提升到 10.804 dB。",
  },
  target: {
    title: "目标声源输出",
    body: "最终输出仅包含目标说话人的单路语音。报告中的指标、参考语音探针和 demo 音频均围绕这一输出进行评估。",
  },
};
function pctGain(value, base) {
  return ((value - base) / base) * 100;
}

function pctReduction(value, base) {
  return ((base - value) / base) * 100;
}

function dbGain(value, base) {
  return value - base;
}

function canvas(id) {
  const el = document.getElementById(id);
  return [el, el.getContext("2d")];
}

function clear(ctx, w, h) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = palette.bg;
  ctx.fillRect(0, 0, w, h);
}

function text(ctx, value, x, y, size = 16, color = "#eef5f8", weight = 600) {
  ctx.fillStyle = color;
  ctx.font = `${weight} ${size}px Segoe UI, Microsoft YaHei, sans-serif`;
  ctx.fillText(value, x, y);
}

function centerText(ctx, value, x, y, size = 16, color = "#eef5f8", weight = 600) {
  ctx.fillStyle = color;
  ctx.font = `${weight} ${size}px Segoe UI, Microsoft YaHei, sans-serif`;
  ctx.fillText(value, x - ctx.measureText(value).width / 2, y);
}

function multiline(ctx, value, x, y, maxWidth, lineHeight, size = 14, color = palette.muted, weight = 500) {
  ctx.fillStyle = color;
  ctx.font = `${weight} ${size}px Segoe UI, Microsoft YaHei, sans-serif`;
  const chars = value.split("");
  let lineText = "";
  let lineY = y;
  chars.forEach((char) => {
    const test = lineText + char;
    if (ctx.measureText(test).width > maxWidth && lineText) {
      ctx.fillText(lineText, x, lineY);
      lineText = char;
      lineY += lineHeight;
    } else {
      lineText = test;
    }
  });
  if (lineText) ctx.fillText(lineText, x, lineY);
}

function line(ctx, x1, y1, x2, y2, color = palette.cyan, width = 2) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}

function roundRect(ctx, x, y, w, h, r, fill, stroke = palette.line) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 1;
  ctx.stroke();
}

function seeded(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967295;
  };
}

function drawWave(ctx, x, y, w, h, seed, color, intensity = 1) {
  const rand = seeded(seed);
  const mid = y + h / 2;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < w; i += 3) {
    const t = i / w;
    const env = Math.sin(Math.PI * t) * 0.76 + 0.18;
    const amp = (Math.sin(t * 55) * 0.42 + Math.sin(t * 131) * 0.24 + (rand() - 0.5) * 0.72) * h * 0.34 * env * intensity;
    if (i === 0) ctx.moveTo(x + i, mid + amp);
    else ctx.lineTo(x + i, mid + amp);
  }
  ctx.stroke();
}

function drawArchitecture() {
  const [el, ctx] = canvas("architectureCanvas");
  const { width: w, height: h } = el;
  clear(ctx, w, h);
  const nodes = [
    ["混合语音", "2-3 人 + 噪声", 70, 210, 180, 112, palette.amber],
    ["STFT 编码", "波形转时频图", 335, 210, 164, 112, palette.cyan],
    ["EGSP 探针", "频域负结果", 585, 120, 190, 116, palette.green],
    ["说话人条件", "目标身份绑定", 585, 340, 190, 116, palette.cyan],
    ["TD-SpeakerBeam", "条件分离主干", 895, 210, 220, 112, palette.cyan],
    ["多参考聚合", "embedding pooling", 1210, 120, 190, 116, palette.green],
    ["目标语音", "单路干净输出", 1210, 340, 190, 116, palette.green],
  ];
  nodes.forEach(([label, sub, x, y, bw, bh, color]) => {
    roundRect(ctx, x, y, bw, bh, 12, "rgba(18,24,30,0.94)", color);
    text(ctx, label, x + 18, y + 42, 24, "#eef5f8", 800);
    text(ctx, sub, x + 18, y + 76, 15, palette.muted, 600);
  });
  [
    [250, 266, 335, 266], [499, 266, 585, 178], [499, 266, 585, 398],
    [775, 178, 895, 266], [775, 398, 895, 266], [1115, 266, 1210, 178],
    [1115, 266, 1210, 398], [1305, 236, 1305, 340],
  ].forEach((p) => line(ctx, ...p, palette.cyan, 4));
  drawWave(ctx, 78, 95, 1260, 64, 31, "rgba(255,207,119,0.74)", 0.85);
  drawWave(ctx, 78, 525, 1260, 64, 41, "rgba(98,227,255,0.74)", 0.65);
  text(ctx, "TD-SpeakerBeam 目标说话人提取流程", 70, 56, 28, "#eef5f8", 800);
  text(ctx, "参考语音提供身份线索；错误参考和多参考聚合用于验证目标选择机制", 70, 88, 16, palette.muted, 500);
}

function drawCaseSpectrum(ctx, x, y, w, h, profile, row, rowIndex) {
  const rand = seeded(profile.seed + rowIndex * 97);
  const barCount = 30;
  for (let b = 0; b < barCount; b += 1) {
    const t = b / (barCount - 1);
    const peakBoost = profile.peaks.reduce((sum, peak) => {
      const dist = Math.abs(t - peak);
      return sum + Math.max(0, 1 - dist / 0.075) * 0.56;
    }, 0);
    const rolloff = 1.16 - t * 0.62 + row.bias;
    const jitter = 0.82 + rand() * row.noise;
    const bh = Math.min(h, (18 + h * 0.40 * (rolloff + peakBoost) * jitter) * row.intensity);
    ctx.fillStyle = row.color + (rowIndex === 2 ? "88" : "66");
    ctx.fillRect(x + b * (w / barCount), y + h - bh, Math.max(12, w / barCount - 13), bh);
  }
}

function drawCase(caseKey = "case01") {
  const [el, ctx] = canvas("caseCanvas");
  const { width: w, height: h } = el;
  const profile = caseProfiles[caseKey] || caseProfiles.case01;
  clear(ctx, w, h);
  text(ctx, profile.title, 34, 36, 19, "#eef5f8", 800);
  multiline(ctx, profile.note, 34, 62, 520, 19, 13, palette.muted, 500);
  text(ctx, `SDR ${profile.sdr.toFixed(2)} dB`, 824, 36, 15, palette.green, 800);
  text(ctx, `SI-SDRi ${profile.sdri.toFixed(2)} dB`, 824, 62, 15, palette.cyan, 800);

  profile.rows.forEach((row, idx) => {
    const y = 102 + idx * 112;
    text(ctx, row.label, 34, y + 22, 17, row.color, 800);
    drawWave(ctx, 164, y, 560, 72, profile.seed + idx * 41, row.color, row.intensity * profile.density);
    drawCaseSpectrum(ctx, 760, y + 4, 280, 76, profile, row, idx);
    line(ctx, 164, y + 86, 1040, y + 86, "rgba(38,51,64,0.70)", 1);
  });
}

function drawBarChart() {
  const [el, ctx] = canvas("barChart");
  const { width: w, height: h } = el;
  clear(ctx, w, h);
  const data = [
    ["Open SB", 10.34, palette.muted],
    ["单参考学生", 10.591, palette.amber],
    ["2-enroll", 10.799, palette.green],
  ];
  text(ctx, "SDR 主指标：可靠主干 + embedding pooling 稳定超过 10 dB", 34, 42, 18, palette.muted, 600);
  data.forEach(([label, value, color], i) => {
    const x = 90 + i * 190;
    const bh = Math.max(8, (value - 9.8) / 1.2 * 230);
    ctx.fillStyle = color;
    ctx.fillRect(x, 350 - bh, 88, bh);
    text(ctx, value.toFixed(3), x, 330 - bh, 18, color, 800);
    text(ctx, label, x - 8, 380, 14, "#eef5f8", 600);
  });
}

function drawRadar() {
  const [el, ctx] = canvas("radarChart");
  const { width: w, height: h } = el;
  clear(ctx, w, h);
  const cx = w / 2;
  const cy = h / 2 + 8;
  const axes = ["SI-SDR", "SIR", "SAR", "STOI", "PESQ"];
  const values = [10.804 / 12.52, 23.224 / 27.05, 11.517 / 12.0, 0.546 / 0.60, 1.313 / 1.5];
  text(ctx, "综合能力雷达：失真、干扰抑制和感知指标共同评估", 28, 34, 15, palette.muted, 600);
  for (let r = 1; r <= 4; r += 1) {
    ctx.strokeStyle = "rgba(98,227,255,0.16)";
    ctx.beginPath();
    axes.forEach((_, i) => {
      const a = -Math.PI / 2 + i * Math.PI * 2 / axes.length;
      const x = cx + Math.cos(a) * r * 38;
      const y = cy + Math.sin(a) * r * 38;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();
  }
  ctx.beginPath();
  values.forEach((v, i) => {
    const a = -Math.PI / 2 + i * Math.PI * 2 / axes.length;
    const x = cx + Math.cos(a) * v * 160;
    const y = cy + Math.sin(a) * v * 160;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.fillStyle = "rgba(143,240,176,0.22)";
  ctx.fill();
  ctx.strokeStyle = palette.green;
  ctx.stroke();
  axes.forEach((axis, i) => {
    const a = -Math.PI / 2 + i * Math.PI * 2 / axes.length;
    text(ctx, axis, cx + Math.cos(a) * 178 - 22, cy + Math.sin(a) * 178, 12, palette.muted, 600);
  });
}

function drawLineChart() {
  const [el, ctx] = canvas("lineChart");
  const { width: w, height: h } = el;
  clear(ctx, w, h);
  const points = [[0.00, 10.586], [0.02, 10.464], [0.05, 10.348], [0.10, 10.288], [0.20, 10.186], [0.30, 10.079]];
  line(ctx, 70, 340, 690, 340, palette.line, 1);
  line(ctx, 70, 60, 70, 340, palette.line, 1);
  ctx.strokeStyle = palette.cyan;
  ctx.lineWidth = 3;
  ctx.beginPath();
  points.forEach(([beta, sdr], i) => {
    const x = 70 + i * 105;
    const y = 320 - (sdr - 10.0) / 0.65 * 230;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  points.forEach(([beta, sdr], i) => {
    const x = 70 + i * 105;
    const y = 320 - (sdr - 10.0) / 0.65 * 230;
    ctx.fillStyle = palette.green;
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fill();
    text(ctx, beta.toFixed(2), x - 18, 370, 13, palette.muted, 600);
    text(ctx, sdr.toFixed(2), x - 14, y - 14, 13, palette.green, 700);
  });
  text(ctx, "EGSP 推理探针：strength 越大，SI-SDR 越低", 42, 36, 18, palette.muted, 600);
}

function drawEfficiency() {
  const [el, ctx] = canvas("effChart");
  const { width: w, height: h } = el;
  clear(ctx, w, h);
  const rings = [
    ["Small", 8.16 / 12.52, "8.16 dB", palette.muted],
    ["Mid", 10.43 / 12.52, "10.43 dB", palette.cyan],
    ["Teacher", 1.0, "12.52 dB", palette.amber],
  ];
  text(ctx, "容量对照：更大分离主干显著提高目标提取质量", 34, 40, 18, palette.muted, 600);
  rings.forEach(([label, v, note, color], i) => {
    const x = 150 + i * 210;
    const y = 194;
    ctx.strokeStyle = "rgba(38,51,64,0.95)";
    ctx.lineWidth = 20;
    ctx.beginPath();
    ctx.arc(x, y, 70, 0, Math.PI * 2);
    ctx.stroke();
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 70, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * v);
    ctx.stroke();
    centerText(ctx, label, x, y + 6, 18, "#eef5f8", 800);
    centerText(ctx, note, x, y + 112, 14, color, 800);
  });
}

function drawStress() {
  const [el, ctx] = canvas("stressChart");
  const { width: w, height: h } = el;
  clear(ctx, w, h);
  const data = [
    ["数据泄漏", 1.06, 10.32, palette.cyan],
    ["从零蒸馏", 10.43, 10.26, palette.red],
    ["EGSP", 10.586, 10.079, palette.red],
  ];
  text(ctx, "失败案例：问题定位比堆叠模块更重要", 40, 42, 18, palette.muted, 600);
  data.forEach(([label, before, after, color], i) => {
    const x = 96 + i * 360;
    ctx.fillStyle = color;
    ctx.fillRect(x, 350 - before * 18, 86, before * 18);
    ctx.fillStyle = color + "88";
    ctx.fillRect(x + 110, 350 - after * 18, 86, after * 18);
    text(ctx, label, x - 8, 382, 15, "#eef5f8", 700);
    text(ctx, before.toFixed(2), x - 6, 350 - before * 18 - 12, 13, color, 700);
    text(ctx, after.toFixed(2), x + 94, 350 - after * 18 - 12, 13, color, 700);
  });
}

function drawTrainChart() {
  const [el, ctx] = canvas("trainChart");
  const { width: w, height: h } = el;
  clear(ctx, w, h);
  const values = [10.43, 10.26, 10.59, 10.557, 10.577, 10.804];
  line(ctx, 64, 338, 700, 338, palette.line, 1);
  line(ctx, 64, 60, 64, 338, palette.line, 1);
  ctx.strokeStyle = palette.green;
  ctx.lineWidth = 3;
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = 64 + i * (620 / (values.length - 1));
    const y = 338 - (v - 10.2) / 0.7 * 250;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  text(ctx, "训练诊断：蒸馏和 EGSP 均需与监督基线公平比较", 34, 36, 18, palette.muted, 600);
  text(ctx, "顺序：mid supervised / from-scratch distill / fine-tune / EGSP .02 / EGSP .05 / pooling", 34, 386, 13, palette.muted, 500);
}

function drawDifficultyChart() {
  const [el, ctx] = canvas("difficultyChart");
  const { width: w, height: h } = el;
  clear(ctx, w, h);
  const data = [
    ["k=1", 10.586, 0.544],
    ["k=2", 10.804, 0.546],
    ["k=4", 10.684, 0.545],
    ["k=8", 10.662, 0.545],
  ];
  text(ctx, "多参考聚合：k=2 最优，继续增加参考没有继续提升", 34, 40, 18, palette.muted, 600);
  data.forEach(([label, sisdr, stoi], i) => {
    const x = 80 + i * 150;
    const sisdrH = (sisdr - 10.4) * 520;
    const stoiH = stoi * 420;
    ctx.fillStyle = palette.cyan;
    ctx.fillRect(x, 350 - sisdrH, 44, sisdrH);
    ctx.fillStyle = palette.green;
    ctx.fillRect(x + 58, 350 - stoiH, 44, stoiH);
    text(ctx, label, x + 20, 382, 14, "#eef5f8", 700);
    text(ctx, "SI-SDR", x - 4, 350 - sisdrH - 10, 12, palette.cyan, 700);
    text(ctx, "STOI", x + 58, 350 - stoiH - 10, 12, palette.green, 700);
  });
}

function drawRobustChart() {
  const [el, ctx] = canvas("robustChart");
  const { width: w, height: h } = el;
  clear(ctx, w, h);
  const data = [
    ["correct", 10.586, palette.green],
    ["short 1s", 9.191, palette.amber],
    ["noisy 5 dB", 7.318, palette.amber],
    ["shuffled", -7.318, palette.red],
    ["interferer", -23.493, palette.red],
  ];
  text(ctx, "参考语音鲁棒性：错误身份比噪声更致命", 34, 42, 18, palette.muted, 600);
  data.forEach(([label, v, color], i) => {
    const y = 90 + i * 70;
    ctx.fillStyle = "rgba(38,51,64,0.75)";
    ctx.fillRect(160, y, 470, 22);
    ctx.fillStyle = color;
    ctx.fillRect(160, y, Math.max(8, (v + 24) / 36 * 470), 22);
    text(ctx, label, 36, y + 17, 14, "#eef5f8", 700);
    text(ctx, `${v.toFixed(2)} dB`, 648, y + 17, 13, color, 800);
  });
}

function drawWaterfallChart() {
  const [el, ctx] = canvas("waterfallChart");
  const { width: w, height: h } = el;
  clear(ctx, w, h);
  const steps = [
    ["自写原型", 1.06, palette.red],
    ["Open SB", 10.32, palette.cyan],
    ["Mid student", 10.43, palette.green],
    ["Distill FT", 10.59, palette.amber],
    ["2-enroll", 10.80, palette.green],
  ];
  text(ctx, "系统诊断路径：可靠主干和参考验证带来可信结果", 34, 42, 18, palette.muted, 600);
  steps.forEach(([label, v, color], i) => {
    const x = 48 + i * 130;
    const y = 340 - v / 12 * 260;
    roundRect(ctx, x, y, 100, 340 - y, 6, color + "aa", color);
    text(ctx, label, x - 10, 374, 13, "#eef5f8", 700);
    text(ctx, v.toFixed(2), x + 18, y - 12, 14, color, 800);
    if (i < steps.length - 1) line(ctx, x + 100, y, x + 130, 340 - steps[i + 1][1] / 12 * 260, palette.cyan, 2);
  });
}

function drawCognition() {
  const [el, ctx] = canvas("cognitionCanvas");
  const { width: w, height: h } = el;
  clear(ctx, w, h);
  const steps = [
    ["目标设定", "参考语音确定要听谁", palette.amber],
    ["对象绑定", "音色与时频线索合并", palette.cyan],
    ["频域探针", "EGSP 负结果提醒训练一致性", palette.green],
    ["条件分离", "TD-SpeakerBeam 抑制干扰", palette.cyan],
    ["容量预算", "学生/教师模型对照", palette.red],
    ["实验复核", "错误参考检查身份依赖", palette.red],
  ];
  steps.forEach(([title, desc, color], i) => {
    const x = 48 + i * 204;
    roundRect(ctx, x, 112, 166, 176, 8, "rgba(18,24,30,0.94)", color);
    text(ctx, title, x + 16, 168, 18, color, 800);
    multiline(ctx, desc, x + 16, 204, 124, 20, 13, palette.muted, 500);
    if (i < steps.length - 1) line(ctx, x + 166, 200, x + 204, 200, palette.cyan, 3);
  });
  text(ctx, "人类听觉注意并非单点选择，而是目标、绑定、抑制和反馈共同工作的循环", 48, 56, 18, palette.muted, 600);
  drawWave(ctx, 98, 334, 1040, 48, 88, "rgba(255,207,119,0.58)", 0.98);
  drawWave(ctx, 98, 392, 1040, 48, 128, "rgba(98,227,255,0.82)", 0.72);
  drawWave(ctx, 98, 450, 1040, 48, 168, "rgba(143,240,176,0.66)", 0.54);
  text(ctx, "混合输入", 48, 364, 13, palette.amber, 700);
  text(ctx, "注意聚焦", 48, 422, 13, palette.cyan, 700);
  text(ctx, "目标输出", 48, 480, 13, palette.green, 700);
}

function renderAll() {
  drawArchitecture();
  drawCase(document.getElementById("caseSelect")?.value || "case01");
  drawBarChart();
  drawRadar();
  drawLineChart();
  drawEfficiency();
  drawTrainChart();
  drawDifficultyChart();
  drawRobustChart();
  drawWaterfallChart();
  drawStress();
  drawCognition();
}

function audioFile(caseKey, track) {
  return `../demo_audio/${cases[caseKey]}_${track}.wav`;
}

function setCase(caseKey) {
  document.querySelectorAll(".audio-card").forEach((card) => {
    const track = card.dataset.track;
    card.querySelector("audio").src = audioFile(caseKey, track);
  });
  drawCase(caseKey);
}

function stopAudio() {
  document.querySelectorAll("audio").forEach((audio) => {
    audio.pause();
    audio.currentTime = 0;
  });
}

document.querySelectorAll(".nav-item").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((item) => item.classList.remove("active"));
    document.querySelectorAll(".view").forEach((view) => view.classList.remove("active"));
    button.classList.add("active");
    document.getElementById(button.dataset.view).classList.add("active");
    renderAll();
  });
});

document.querySelectorAll(".play-btn").forEach((button) => {
  button.addEventListener("click", () => {
    stopAudio();
    button.parentElement.querySelector("audio").play();
  });
});

document.getElementById("caseSelect").addEventListener("change", (event) => setCase(event.target.value));
window.addEventListener("resize", renderAll);

document.querySelectorAll(".hotspot").forEach((spot) => {
  const show = () => {
    const info = moduleInfo[spot.dataset.module];
    const tooltip = document.getElementById("moduleTooltip");
    tooltip.querySelector("strong").textContent = info.title;
    tooltip.querySelector("p").textContent = info.body;
  };
  spot.addEventListener("mouseenter", show);
  spot.addEventListener("focus", show);
});

setCase("case01");
renderAll();
