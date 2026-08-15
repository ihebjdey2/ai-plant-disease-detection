(() => {
  "use strict";

  const imageInput = document.getElementById("imageInput");
  const preview = document.getElementById("preview");
  const previewWrap = document.getElementById("previewWrap");
  const dropZone = document.getElementById("dropZone");
  const scanForm = document.getElementById("scanForm");
  const scanButton = document.getElementById("scanButton");
  const fileName = document.querySelector("[data-file-name]");
  const changeFile = document.querySelector("[data-change-file]");
  const uploadTitle = document.querySelector("[data-upload-title]");
  let previewUrl = null;

  const showSelectedFile = (file) => {
    if (!file || !preview || !previewWrap) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    previewWrap.hidden = false;
    if (fileName) fileName.textContent = file.name;
    if (changeFile) changeFile.hidden = false;
    if (uploadTitle) uploadTitle.textContent = uploadTitle.dataset.readyText || uploadTitle.textContent;
  };

  if (imageInput) {
    if (uploadTitle) uploadTitle.dataset.readyText = imageInput.dataset.readyText || uploadTitle.textContent;
    imageInput.addEventListener("change", () => showSelectedFile(imageInput.files[0]));
  }

  if (dropZone && imageInput) {
    ["dragenter", "dragover"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.add("is-dragging");
      });
    });
    ["dragleave", "drop"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.remove("is-dragging");
      });
    });
    dropZone.addEventListener("drop", (event) => {
      const files = event.dataTransfer?.files;
      if (!files?.length) return;
      const transfer = new DataTransfer();
      transfer.items.add(files[0]);
      imageInput.files = transfer.files;
      imageInput.dispatchEvent(new Event("change", { bubbles: true }));
    });
  }

  if (scanForm && scanButton) {
    scanForm.addEventListener("submit", () => {
      scanForm.setAttribute("aria-busy", "true");
      scanButton.disabled = true;
      scanButton.classList.add("is-loading");
      const label = scanButton.querySelector("[data-button-label]");
      if (label) label.textContent = scanButton.dataset.analyzingText;
    });
  }

  window.addEventListener("pagehide", () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  });

  document.querySelectorAll("[data-password-toggle]").forEach((toggle) => {
    const input = document.getElementById(toggle.getAttribute("aria-controls"));
    if (!input) return;
    toggle.addEventListener("click", () => {
      const showing = input.type === "text";
      input.type = showing ? "password" : "text";
      const label = showing ? toggle.dataset.showLabel : toggle.dataset.hideLabel;
      toggle.textContent = label;
      toggle.setAttribute("aria-label", label);
      input.focus();
    });
  });

  document.querySelectorAll("form[data-confirm]").forEach((form) => {
    form.addEventListener("submit", (event) => {
      if (!window.confirm(form.dataset.confirm)) event.preventDefault();
    });
  });

  const filterButtons = [...document.querySelectorAll("[data-history-filter]")];
  const historyCards = [...document.querySelectorAll("[data-history-card]")];
  const filterStatus = document.getElementById("filter-status");
  filterButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const selected = button.dataset.historyFilter;
      let visibleCount = 0;
      filterButtons.forEach((item) => {
        const active = item === button;
        item.classList.toggle("is-active", active);
        item.setAttribute("aria-pressed", String(active));
      });
      historyCards.forEach((card) => {
        const visible = selected === "all" || card.dataset.status === selected;
        card.hidden = !visible;
        if (visible) visibleCount += 1;
      });
      if (filterStatus) {
        filterStatus.textContent = `${visibleCount} ${filterStatus.dataset.resultsLabel || ""}`.trim();
      }
    });
  });
})();
