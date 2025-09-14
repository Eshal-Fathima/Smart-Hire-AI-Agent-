document.addEventListener("DOMContentLoaded", () => {
    // Hero animation
    const heroText = document.querySelector(".hero h1");
    const heroSub = document.querySelector(".hero p");
    setTimeout(() => {
        heroText.style.opacity = "1";
        heroText.style.transform = "translateY(0)";
    }, 500);
    setTimeout(() => {
        heroSub.style.opacity = "1";
    }, 1200);
});

document.addEventListener("DOMContentLoaded", () => {
  const ctx = document.getElementById("skillsChart").getContext("2d");

  new Chart(ctx, {
    type: "pie",
    data: {
      labels: ["Correct Skills", "Missing Skills"],
      datasets: [
        {
          data: [correctSkills, missingSkills],
          backgroundColor: ["#4caf50", "#f44336"],
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            font: { size: 14 }
          }
        }
      }
    },
  });
});
