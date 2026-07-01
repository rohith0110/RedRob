import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { Presentation, PresentationFile } from "@oai/artifact-tool";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CANVAS = "#FFFFFF";
const INK = "#111111";
const MUTED = "#555555";
const PANEL = "#EDEDED";
const RULE = "#B8BCC4";
const HIGHLIGHT = "#FF6B35";
const FONT = "Helvetica Neue";

const PPTX_PATH = path.join(__dirname, "redrob_candidate_ranking_system.pptx");
const MONTAGE_PATH = path.join(__dirname, "assets", "deck_montage.webp");
const SCREEN_HOME = path.join(__dirname, "assets", "sandbox_home_cropped.png");
const SCREEN_RANKED = path.join(__dirname, "assets", "sandbox_ranked_cropped.png");

function slideFrame() {
  return { left: 56, top: 46, width: 1168, height: 628 };
}

async function writeBlob(targetPath, blob) {
  await fs.mkdir(path.dirname(targetPath), { recursive: true });
  await fs.writeFile(targetPath, new Uint8Array(await blob.arrayBuffer()));
}

async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function addText(slide, options) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: {
      left: options.left,
      top: options.top,
      width: options.width,
      height: options.height,
    },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = options.text;
  shape.text.style = {
    fontFace: FONT,
    fontSize: options.fontSize,
    bold: Boolean(options.bold),
    color: options.color ?? INK,
    alignment: options.alignment ?? "left",
  };
  return shape;
}

function addPanel(slide, options) {
  return slide.shapes.add({
    geometry: options.geometry ?? "rect",
    position: {
      left: options.left,
      top: options.top,
      width: options.width,
      height: options.height,
    },
    fill: options.fill ?? PANEL,
    line: { style: "solid", fill: options.lineFill ?? PANEL, width: options.lineWidth ?? 0 },
    borderRadius: options.borderRadius,
  });
}

function addRule(slide, left, top, width, height = 2, fill = RULE) {
  addPanel(slide, { left, top, width, height, fill, lineFill: fill });
}

function addKicker(slide, text) {
  addText(slide, { left: 56, top: 28, width: 220, height: 18, text, fontSize: 14, color: MUTED, bold: true });
}

function addTitle(slide, title, subtitle) {
  addText(slide, { left: 56, top: 78, width: 760, height: 132, text: title, fontSize: 38, bold: true });
  if (subtitle) {
    addText(slide, { left: 56, top: 208, width: 760, height: 72, text: subtitle, fontSize: 20, color: MUTED });
  }
}

function addFooter(slide, pageNumber) {
  addText(slide, { left: 1160, top: 680, width: 60, height: 20, text: String(pageNumber), fontSize: 13, color: MUTED, alignment: "right" });
}

function addMetricCard(slide, options) {
  addPanel(slide, { left: options.left, top: options.top, width: options.width, height: options.height, fill: PANEL });
  addText(slide, { left: options.left + 18, top: options.top + 16, width: options.width - 36, height: 52, text: options.value, fontSize: 32, bold: true, color: options.color ?? INK });
  addText(slide, { left: options.left + 18, top: options.top + 70, width: options.width - 36, height: 54, text: options.label, fontSize: 17, color: MUTED });
}

function bulletBlock(lines) {
  return lines.map((line) => `• ${line}`).join("\n");
}

function addThreeColumnCards(slide, cards, top = 290) {
  const cardWidth = 354;
  const gutter = 28;
  cards.forEach((card, index) => {
    const left = 56 + index * (cardWidth + gutter);
    addPanel(slide, { left, top, width: cardWidth, height: 260, fill: PANEL });
    addText(slide, { left: left + 20, top: top + 18, width: cardWidth - 40, height: 44, text: card.title, fontSize: 24, bold: true });
    addText(slide, { left: left + 20, top: top + 78, width: cardWidth - 40, height: 158, text: bulletBlock(card.lines), fontSize: 18, color: MUTED });
  });
}

function addPipelineBox(slide, left, top, width, height, title, body) {
  addPanel(slide, { left, top, width, height, fill: PANEL });
  addText(slide, { left: left + 14, top: top + 14, width: width - 28, height: 28, text: title, fontSize: 20, bold: true });
  addText(slide, { left: left + 14, top: top + 48, width: width - 28, height: height - 60, text: body, fontSize: 16, color: MUTED });
}

function addArrowText(slide, left, top) {
  addText(slide, { left, top, width: 26, height: 28, text: "→", fontSize: 22, bold: true, color: MUTED, alignment: "center" });
}

function addScreenshot(slide, bytes, left, top, width, height, alt) {
  slide.images.add({
    blob: bytes,
    contentType: "image/png",
    alt,
    fit: "contain",
    position: { left, top, width, height },
    geometry: "rect",
  });
  addPanel(slide, { left, top, width, height, fill: "none", lineFill: RULE, lineWidth: 1 });
}

function populateSlides(presentation, images) {
  const slide1 = presentation.slides.add();
  slide1.background.fill = CANVAS;
  addKicker(slide1, "Redrob Step 3B Release");
  addTitle(
    slide1,
    "Beyond Keywords",
    "Evidence-based candidate ranking for the Senior AI Engineer shortlist"
  );
  addRule(slide1, 56, 284, 1168, 2, HIGHLIGHT);
  addMetricCard(slide1, { left: 56, top: 330, width: 220, height: 122, value: "100k", label: "candidate records processed offline" });
  addMetricCard(slide1, { left: 302, top: 330, width: 220, height: 122, value: "CPU", label: "no GPU and no hosted API" });
  addMetricCard(slide1, { left: 548, top: 330, width: 220, height: 122, value: "226.9s", label: "final local frozen-tag runtime", color: HIGHLIGHT });
  addText(slide1, {
    left: 56,
    top: 500,
    width: 710,
    height: 118,
    text: "The release keeps ranking logic frozen, regenerates the final CSV from the tagged code path, validates the top 100 explanations, and reproduces the same shortlist offline in Docker.",
    fontSize: 22,
    color: MUTED,
  });
  addPanel(slide1, { left: 860, top: 314, width: 320, height: 250, fill: PANEL });
  addText(slide1, { left: 884, top: 346, width: 272, height: 42, text: "Central release claim", fontSize: 24, bold: true });
  addText(slide1, {
    left: 884,
    top: 400,
    width: 260,
    height: 120,
    text: "Career evidence outranks skill lists, explanations stay grounded, and output order remains deterministic: score descending, then candidate_id ascending.",
    fontSize: 20,
    color: MUTED,
  });
  addFooter(slide1, 1);

  const slide2 = presentation.slides.add();
  slide2.background.fill = CANVAS;
  addKicker(slide2, "Why keyword ranking fails");
  addTitle(slide2, "Keyword filters blur real ranking depth with copied buzzwords");
  addText(slide2, {
    left: 56,
    top: 188,
    width: 1100,
    height: 54,
    text: "The challenge is not counting AI terms. It is separating candidates who built ranking systems from candidates who only mention them.",
    fontSize: 22,
    color: MUTED,
  });
  addThreeColumnCards(slide2, [
    { title: "Skill-list inflation", lines: ["Raw skills are easy to copy", "Advanced IR terms can appear without supporting work history", "Uncorroborated claims should not dominate ranking"] },
    { title: "Title ambiguity", lines: ["Senior titles can hide weak retrieval depth", "Research-heavy roles may not imply shipped ranking systems", "Descriptions matter more than labels"] },
    { title: "Recruiter noise", lines: ["Availability matters, but only within bounds", "Sparse profiles need conservative explanations", "Determinism matters for reviewer trust"] },
  ]);
  addFooter(slide2, 2);

  const slide3 = presentation.slides.add();
  slide3.background.fill = CANVAS;
  addKicker(slide3, "What the role requires");
  addTitle(slide3, "The shortlist favors retrieval, ranking, evaluation, and product delivery evidence");
  addThreeColumnCards(slide3, [
    { title: "Retrieval and ranking depth", lines: ["Search relevance", "Recommendation systems", "Matching engines", "Learning-to-rank or ranking ownership"] },
    { title: "Evaluation and shipping", lines: ["Offline metrics or A/B tests", "Production deployment", "Monitoring and iteration", "User-facing impact"] },
    { title: "Practical engineering", lines: ["Python services and APIs", "Data or search infrastructure", "Recent coding evidence", "Cross-functional product execution"] },
  ], 260);
  addText(slide3, { left: 56, top: 588, width: 1100, height: 42, text: "This keeps the role definition tied to shipped systems instead of broad AI adjacency.", fontSize: 18, color: MUTED });
  addFooter(slide3, 3);

  const slide4 = presentation.slides.add();
  slide4.background.fill = CANVAS;
  addKicker(slide4, "Evidence hierarchy");
  addTitle(slide4, "Career history outranks summaries, and summaries outrank skill lists");
  const rows = [
    ["1", "career_history.description", "Highest-signal proof of what the candidate actually built"],
    ["2", "career_history.title", "Useful only when paired with the description"],
    ["3", "profile.summary / headline", "Good support, weaker than work history"],
    ["4", "skills / assessments", "Lowest-trust source unless corroborated"],
  ];
  rows.forEach((row, index) => {
    const top = 214 + index * 92;
    addPanel(slide4, { left: 56, top, width: 1168, height: 72, fill: index % 2 === 0 ? PANEL : "#F7F7F7" });
    addText(slide4, { left: 78, top: top + 18, width: 52, height: 28, text: row[0], fontSize: 22, bold: true, color: HIGHLIGHT });
    addText(slide4, { left: 142, top: top + 18, width: 330, height: 28, text: row[1], fontSize: 20, bold: true });
    addText(slide4, { left: 500, top: top + 14, width: 680, height: 40, text: row[2], fontSize: 18, color: MUTED });
  });
  addText(slide4, { left: 56, top: 610, width: 1120, height: 44, text: "This ordering is the main defense against resumes that list modern tools without showing ranking-relevant ownership.", fontSize: 18, color: MUTED });
  addFooter(slide4, 4);

  const slide5 = presentation.slides.add();
  slide5.background.fill = CANVAS;
  addKicker(slide5, "System architecture");
  addTitle(slide5, "The pipeline persists every stage and only resumes when fingerprints still match");
  const pipelineRows = [
    [
      ["Audit", "schema and reference-date checks"],
      ["Normalize", "candidate context and text prep"],
      ["Evidence", "plain-language ledger extraction"],
      ["Credibility", "contradiction and support checks"],
    ],
    [
      ["Behavioral", "availability and logistics bounds"],
      ["Score", "deterministic component ranking"],
      ["Reasoning", "grounded top-100 explanations"],
      ["Submit", "validated submission writer"],
    ],
  ];
  pipelineRows.forEach((row, rowIndex) => {
    row.forEach(([title, body], columnIndex) => {
      const left = 56 + columnIndex * 286;
      const top = 256 + rowIndex * 158;
      addPipelineBox(slide5, left, top, 250, 116, title, body);
      if (columnIndex < row.length - 1) {
        addArrowText(slide5, left + 255, top + 38);
      }
    });
  });
  addText(slide5, { left: 56, top: 584, width: 1120, height: 54, text: "Each stage writes artifacts, hashes, and elapsed times under runs/<run_id>. Resume mode skips only when input, config, and upstream fingerprints remain unchanged.", fontSize: 18, color: MUTED });
  addFooter(slide5, 5);

  const slide6 = presentation.slides.add();
  slide6.background.fill = CANVAS;
  addKicker(slide6, "Evidence-first matching");
  addTitle(slide6, "Plain language matters more than exact buzzwords");
  addThreeColumnCards(slide6, [
    { title: "Recognized positive evidence", lines: ["search relevance", "recommendation systems", "personalized feed", "marketplace ranking"] },
    { title: "Operational evidence", lines: ["A/B evaluation", "production deployment", "monitoring", "Python services"] },
    { title: "Scoring summary", lines: ["base_fit_score", "× credibility_multiplier", "× availability_multiplier", "× 100"] },
  ], 270);
  addText(slide6, { left: 56, top: 598, width: 1120, height: 44, text: "A candidate can rank well by describing real ranking or retrieval work even if their wording is not a perfect keyword match.", fontSize: 18, color: MUTED });
  addFooter(slide6, 6);

  const slide7 = presentation.slides.add();
  slide7.background.fill = CANVAS;
  addKicker(slide7, "Defenses");
  addTitle(slide7, "Credibility and corroboration defenses keep copied AI language from dominating");
  addThreeColumnCards(slide7, [
    { title: "Corroboration", lines: ["Skills alone are weak evidence", "Work history must support advanced claims", "Career evidence can outweigh keyword-heavy profiles"] },
    { title: "Contradictions", lines: ["Title and description mismatches lower credibility", "Sparse support reduces trust", "Penalties are explicit and deterministic"] },
    { title: "Bounded modifiers", lines: ["Availability range is 0.72 to 1.08", "Behavioral signals separate close candidates", "They do not rescue weak technical fit"] },
  ], 276);
  addFooter(slide7, 7);

  const slide8 = presentation.slides.add();
  slide8.background.fill = CANVAS;
  addKicker(slide8, "Availability and logistics");
  addTitle(slide8, "Availability helps shortlist triage, but only inside a narrow multiplier band");
  addPanel(slide8, { left: 100, top: 318, width: 980, height: 40, fill: PANEL });
  addPanel(slide8, { left: 352, top: 318, width: 420, height: 40, fill: HIGHLIGHT, lineFill: HIGHLIGHT });
  addText(slide8, { left: 88, top: 274, width: 84, height: 26, text: "0.72", fontSize: 24, bold: true });
  addText(slide8, { left: 1110, top: 274, width: 84, height: 26, text: "1.08", fontSize: 24, bold: true, alignment: "right" });
  addText(slide8, { left: 412, top: 364, width: 300, height: 28, text: "shortlist triage zone", fontSize: 20, bold: true, color: HIGHLIGHT, alignment: "center" });
  addThreeColumnCards(slide8, [
    { title: "Signals used", lines: ["notice period", "last-active recency", "location logistics"] },
    { title: "What it changes", lines: ["ordering among similar candidates", "recruiter urgency", "practical interview viability"] },
    { title: "What it cannot do", lines: ["override weak ranking evidence", "invent missing experience", "replace human review"] },
  ], 430);
  addFooter(slide8, 8);

  const slide9 = presentation.slides.add();
  slide9.background.fill = CANVAS;
  addKicker(slide9, "Explainability");
  addTitle(slide9, "The release explanation layer changed wording without changing scores or shortlist order");
  addMetricCard(slide9, { left: 56, top: 240, width: 220, height: 122, value: "100/100", label: "top-100 style lint passed" });
  addMetricCard(slide9, { left: 300, top: 240, width: 220, height: 122, value: "100/100", label: "top-100 grounding passed" });
  addMetricCard(slide9, { left: 544, top: 240, width: 220, height: 122, value: "0", label: "first-person recruiter leaks in final run", color: HIGHLIGHT });
  addMetricCard(slide9, { left: 788, top: 240, width: 220, height: 122, value: "true", label: "top-100 IDs, ranks, scores unchanged" });
  addText(slide9, {
    left: 56,
    top: 408,
    width: 1120,
    height: 90,
    text: "The Step 3B patch only neutralized first-person source phrasing such as 'our product' into recruiter-facing narration. The final local release run preserved the approved score CSV hash and the exact top-100 candidate IDs, ranks, and scores.",
    fontSize: 21,
    color: MUTED,
  });
  addText(slide9, { left: 56, top: 548, width: 1120, height: 54, text: "Reasoning is selected from source-grounded evidence IDs, linted, and then checked again for grounding before release.", fontSize: 18, color: MUTED });
  addFooter(slide9, 9);

  const slide10 = presentation.slides.add();
  slide10.background.fill = CANVAS;
  addKicker(slide10, "Runtime and reproducibility");
  addTitle(slide10, "The release stays offline and reproducible across local and Docker runs");
  slide10.charts.add("bar", {
    position: { left: 56, top: 246, width: 520, height: 300 },
    categories: ["Local", "Docker 1", "Docker 2"],
    series: [{ name: "Seconds", values: [226.949, 247.042, 251.11], fill: "accent1" }],
    hasLegend: false,
    dataLabels: { showValue: true, position: "outEnd" },
    yAxis: { majorGridlines: { style: "solid", fill: RULE, width: 1 } },
  });
  addMetricCard(slide10, { left: 654, top: 246, width: 250, height: 122, value: "3982 MB", label: "local peak memory" });
  addMetricCard(slide10, { left: 930, top: 246, width: 250, height: 122, value: "7132 MB", label: "Docker verification peak memory", color: HIGHLIGHT });
  addMetricCard(slide10, { left: 654, top: 392, width: 250, height: 122, value: "false", label: "hosted API during ranking" });
  addMetricCard(slide10, { left: 930, top: 392, width: 250, height: 122, value: "true", label: "local and Docker submission hash match" });
  addText(slide10, { left: 56, top: 574, width: 1120, height: 52, text: "Docker remained under the official 300-second limit but inside the project warning band, so a second network-disabled verification run was executed and matched the local output.", fontSize: 18, color: MUTED });
  addFooter(slide10, 10);

  const slide11 = presentation.slides.add();
  slide11.background.fill = CANVAS;
  addKicker(slide11, "Sandbox");
  addTitle(slide11, "The recruiter sandbox uses the same ranking modules on small JSON or JSONL samples");
  addScreenshot(slide11, images.ranked, 56, 230, 760, 360, "Sandbox ranked results screen");
  addScreenshot(slide11, images.home, 858, 230, 322, 182, "Sandbox landing screen");
  addText(slide11, { left: 858, top: 438, width: 310, height: 138, text: bulletBlock(["upload JSON or JSONL", "rank up to 100 candidates", "download ranked CSV", "marked as non-official for small samples"]), fontSize: 18, color: MUTED });
  addText(slide11, { left: 56, top: 610, width: 1120, height: 46, text: "The app surfaces rank, score components, evidence excerpts, modifiers, and the same recruiter-facing reasoning generated by the production pipeline.", fontSize: 18, color: MUTED });
  addFooter(slide11, 11);

  const slide12 = presentation.slides.add();
  slide12.background.fill = CANVAS;
  addKicker(slide12, "Limitations and next step");
  addTitle(slide12, "This system supports recruiter triage, not final hiring decisions");
  addThreeColumnCards(slide12, [
    { title: "Limits", lines: ["depends on profile quality", "does not infer missing evidence", "no claims about hiring outcomes"] },
    { title: "Responsible use", lines: ["human review remains required", "top-ranked profiles should be spot-checked", "availability is secondary to role fit"] },
    { title: "Practical next step", lines: ["add recruiter feedback loop after the challenge", "keep offline guarantees", "reassess only with evidence-backed changes"] },
  ], 278);
  addFooter(slide12, 12);
}

export async function buildDeck() {
  const presentation = Presentation.create({
    slideSize: { width: 1280, height: 720 },
  });

  const images = {
    home: await readImageBlob(SCREEN_HOME),
    ranked: await readImageBlob(SCREEN_RANKED),
  };

  populateSlides(presentation, images);

  await fs.mkdir(path.join(__dirname, "assets"), { recursive: true });
  const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
  await writeBlob(MONTAGE_PATH, montage);

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(PPTX_PATH);
  return { pptxPath: PPTX_PATH, montagePath: MONTAGE_PATH };
}

if (typeof process !== "undefined" && process.argv[1] === fileURLToPath(import.meta.url)) {
  const result = await buildDeck();
  console.log(JSON.stringify(result, null, 2));
}
