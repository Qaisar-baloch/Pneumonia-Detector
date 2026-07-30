(function () {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const loadingState = document.getElementById('loadingState');
  const imageState = document.getElementById('imageState');
  const xrayImg = document.getElementById('xrayImg');
  const viewToggle = document.getElementById('viewToggle');
  const resetBtn = document.getElementById('resetBtn');

  const readoutEmpty = document.getElementById('readoutEmpty');
  const readoutContent = document.getElementById('readoutContent');
  const readoutError = document.getElementById('readoutError');

  const verdictLabel = document.getElementById('verdictLabel');
  const gaugeFill = document.getElementById('gaugeFill');
  const gaugeValue = document.getElementById('gaugeValue');
  const confidenceValue = document.getElementById('confidenceValue');
  const focusValue = document.getElementById('focusValue');

  let images = { original: null, heatmap: null };

  if (!dropzone || fileInput.disabled) {
    // Model isn't loaded; upload is disabled server-side.
    return;
  }

  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });

  ['dragover', 'dragenter'].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add('drag-over');
    })
  );
  ['dragleave', 'drop'].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove('drag-over');
    })
  );
  dropzone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
  });

  resetBtn.addEventListener('click', () => {
    dropzone.hidden = false;
    imageState.hidden = true;
    loadingState.hidden = true;
    viewToggle.hidden = true;
    resetBtn.hidden = true;
    readoutContent.hidden = true;
    readoutError.hidden = true;
    readoutEmpty.hidden = false;
    fileInput.value = '';
  });

  viewToggle.querySelectorAll('.toggle-option').forEach((btn) => {
    btn.addEventListener('click', () => {
      viewToggle.querySelectorAll('.toggle-option').forEach((b) => b.classList.remove('is-active'));
      btn.classList.add('is-active');
      const view = btn.dataset.view;
      xrayImg.src = view === 'heatmap' ? images.heatmap : images.original;
    });
  });

  function handleFile(file) {
    dropzone.hidden = true;
    loadingState.hidden = false;
    readoutEmpty.hidden = true;
    readoutError.hidden = true;
    readoutContent.hidden = true;

    const formData = new FormData();
    formData.append('image', file);

    fetch('/predict', { method: 'POST', body: formData })
      .then((res) =>
        res.json().then((data) => {
          if (!res.ok) throw new Error(data.error || 'Something went wrong.');
          return data;
        })
      )
      .then(showResult)
      .catch(showError);
  }

  function showResult(data) {
    images.original = 'data:image/png;base64,' + data.original_image;
    images.heatmap = 'data:image/png;base64,' + data.heatmap_image;

    xrayImg.src = images.original;
    loadingState.hidden = true;
    imageState.hidden = false;
    viewToggle.hidden = false;
    resetBtn.hidden = false;
    viewToggle.querySelectorAll('.toggle-option').forEach((b) => b.classList.remove('is-active'));
    viewToggle.querySelector('[data-view="original"]').classList.add('is-active');

    const isPneumonia = data.prediction === 'PNEUMONIA';
    verdictLabel.textContent = data.prediction;
    verdictLabel.className = 'readout-verdict ' + (isPneumonia ? 'is-pneumonia' : 'is-normal');

    // Gauge: 0 = certain NORMAL, 100 = certain PNEUMONIA
    const pct = data.pneumonia_probability;
    gaugeFill.style.left = pct + '%';
    gaugeValue.textContent = pct + '% pneumonia';

    confidenceValue.textContent = data.confidence + '%';
    focusValue.textContent = data.focus_region;

    readoutContent.hidden = false;
  }

  function showError(err) {
    loadingState.hidden = true;
    dropzone.hidden = false;
    readoutError.textContent = err.message || 'Could not analyze that image.';
    readoutError.hidden = false;
  }
})();
