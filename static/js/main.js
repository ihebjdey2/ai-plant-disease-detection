const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const button = document.getElementById("scanButton");

if (imageInput && preview) {
  imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (file) {
      preview.src = URL.createObjectURL(file);
      preview.style.display = "block";
    }
  });
}

if (button) {
  button.closest("form").addEventListener("submit", () => {
    button.disabled = true;
    button.textContent = button.dataset.analyzingText;
  });
}
