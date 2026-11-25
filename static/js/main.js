/* PulmoScan AI — front-end logic */
(function () {
  const fileInput = document.getElementById("file-input");
  const dropzone = document.getElementById("dropzone");
  const dzEmpty = dropzone.querySelector(".dz-empty");
  const preview = document.getElementById("preview");
  const analyzeBtn = document.getElementById("analyze-btn");
  const resetBtn = document.getElementById("reset-btn");
  const form = document.getElementById("upload-form");
  const btnLabel = analyzeBtn.querySelector(".btn-label");
  const spinner = analyzeBtn.querySelector(".spinner");

  const resultEmpty = document.getElementById("result-empty");
  const resultContent = document.getElementById("result-content");
  const resultError = document.getElementById("result-error");

  let selectedFile = null;

  /* ---------- file selection ---------- */
  function handleFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      preview.src = e.target.result;
      preview.hidden = false;
      dzEmpty.hidden = true;
    };
    reader.readAsDataURL(file);
    analyzeBtn.disabled = false;
    resetBtn.hidden = false;
  }

  fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

  ["dragover", "dragenter"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) { fileInput.files = e.dataTransfer.files; handleFile(file); }
  });

  /* ---------- reset ---------- */
  resetBtn.addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    preview.hidden = true;
    dzEmpty.hidden = false;
    analyzeBtn.disabled = true;
    resetBtn.hidden = true;
    resultContent.hidden = true;
    resultError.hidden = true;
    resultEmpty.hidden = false;
    document.getElementById("guidance").hidden = true;
  });

  /* ---------- loading state ---------- */
  function setLoading(on) {
    analyzeBtn.disabled = on;
    spinner.hidden = !on;
    btnLabel.textContent = on ? "Analyzing…" : "Analyze X-ray";
  }

  /* ---------- submit ---------- */
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!selectedFile) return;
    setLoading(true);
    resultError.hidden = true;

    const data = new FormData();
    data.append("image", selectedFile);

    try {
      const res = await fetch("/predict", { method: "POST", body: data });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || "Prediction failed.");
      renderResult(json);
    } catch (err) {
      resultEmpty.hidden = true;
      resultContent.hidden = true;
      resultError.hidden = false;
      resultError.textContent = "⚠️ " + err.message;
    } finally {
      setLoading(false);
    }
  });

  /* ---------- render ---------- */
  function renderResult(r) {
    resultEmpty.hidden = true;
    resultError.hidden = true;
    resultContent.hidden = false;

    const isTB = r.label === "Tuberculosis";
    const verdict = document.getElementById("verdict");
    verdict.className = "verdict " + (isTB ? "tb" : "normal");
    document.getElementById("verdict-icon").textContent = isTB ? "⚠️" : "✓";
    document.getElementById("verdict-label").textContent = r.label;
    document.getElementById("verdict-confidence").textContent =
      `Confidence: ${r.confidence}%`;

    document.getElementById("normal-pct").textContent = r.normal_probability + "%";
    document.getElementById("tb-pct").textContent = r.tb_probability + "%";
    // animate bars on next frame
    requestAnimationFrame(() => {
      document.getElementById("normal-bar").style.width = r.normal_probability + "%";
      document.getElementById("tb-bar").style.width = r.tb_probability + "%";
    });

    const thrNote = document.getElementById("thr-note");
    if (thrNote && typeof r.threshold === "number") {
      thrNote.textContent =
        `Model raw TB score: ${r.tb_probability}% · flagged as Tuberculosis when ≥ ${r.threshold}% ` +
        `(threshold tuned for screening sensitivity).`;
    }

    document.getElementById("result-img").src = r.image_url;
    const camFig = document.getElementById("cam-figure");
    if (r.gradcam_url) {
      document.getElementById("cam-img").src = r.gradcam_url;
      camFig.hidden = false;
    } else {
      camFig.hidden = true;
    }

    renderGuidance(r.label, r.guidance);
    document.getElementById("guidance").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function fillList(id, items) {
    const ul = document.getElementById(id);
    ul.innerHTML = "";
    (items || []).forEach((t) => {
      const li = document.createElement("li");
      li.textContent = t;
      ul.appendChild(li);
    });
  }

  function renderGuidance(label, g) {
    const section = document.getElementById("guidance");
    document.getElementById("guidance-title").textContent =
      label === "Tuberculosis" ? "Guidance & Remedies" : "Recommendations";
    document.getElementById("guidance-summary").textContent = g.summary;
    fillList("g-immediate", g.immediate_steps);
    fillList("g-treatment", g.treatment);
    fillList("g-lifestyle", g.lifestyle);
    fillList("g-warning", g.warning_signs);
    document.getElementById("g-disclaimer").textContent = g.disclaimer;
    section.hidden = false;
  }
})();
